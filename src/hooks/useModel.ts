import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { getModelSpec, resolveVariant } from '../lib/modelRegistry'
import type { DebugInfo, WorkerRequest, WorkerResponse } from '../workers/protocol'

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'running' | 'error'

export interface RunResult<T = unknown> {
  output: T
  debug?: DebugInfo
  ms: number
}

export interface RunOptions {
  /** Variant cần chạy, với demo khai báo `variants` trong ModelSpec. */
  variantId?: string
  /** Tham số truyền thẳng cho pipeline(), ví dụ { top_k: 5 }. */
  pipelineOptions?: Record<string, unknown>
  /**
   * Tham số VỊ TRÍ chèn giữa input và options — cho các task mà Transformers.js
   * nhận tham số thứ hai theo vị trí:
   *
   *   model.run(imageUrl, { args: [['a cat', 'a dog']] })
   *   -> pipe(imageUrl, ['a cat', 'a dog'], {})
   *
   * Chỉ có tác dụng với mode 'pipeline'.
   */
  args?: unknown[]
  /**
   * Nhận từng mẩu văn bản ngay khi model sinh ra, thay vì chờ xong cả đoạn.
   *
   * Callback này KHÔNG được gửi sang worker (hàm không serialize được) — chỉ
   * việc "có onToken hay không" được gửi đi dưới dạng cờ `stream`, còn callback
   * nằm lại main thread và được tra theo requestId.
   *
   *   await model.run(prompt, { onToken: (t) => setText((prev) => prev + t) })
   */
  onToken?: (text: string) => void
}

/** Số đo theo từng variant — dữ liệu cho phần benchmark của báo cáo. */
export interface VariantTiming {
  loadMs?: number
  lastMs?: number
}

/**
 * Hook đóng gói toàn bộ vòng đời của một model: tạo worker, load, chạy, dọn.
 *
 * Mỗi demo dùng một worker riêng và chỉ load model KHI người dùng bấm — không
 * tự tải lúc mở trang, vì tổng dung lượng mọi model là hàng trăm MB.
 *
 * Với demo có nhiều `variants`, mọi variant dùng chung worker này và được cache
 * riêng, nên đổi qua lại không phải tải lại.
 *
 *   const model = useModel('sentiment')
 *   const res = await model.run<{ labels: string[]; logits: number[] }>(text)
 *
 *   const model = useModel('tokenizer')
 *   const res = await model.run<TokenizeOutput>(text, { variantId: 'gpt2' })
 */
export function useModel(demoId: string) {
  const spec = useMemo(() => getModelSpec(demoId), [demoId])

  const [status, setStatus] = useState<ModelStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [device, setDevice] = useState<'wasm' | 'webgpu' | null>(null)
  const [deviceNote, setDeviceNote] = useState<string | null>(null)
  const [lastMs, setLastMs] = useState<number | null>(null)
  const [loadMs, setLoadMs] = useState<number | null>(null)

  /** Variant đã load xong (theo id). */
  const [readyVariants, setReadyVariants] = useState<string[]>([])
  const [timings, setTimings] = useState<Record<string, VariantTiming>>({})

  /** 0–1, tiến độ tải tổng hợp trên tất cả các file của lần load hiện tại. */
  const [overall, setOverall] = useState(0)
  const [currentFile, setCurrentFile] = useState<string | null>(null)
  const [loadingVariantId, setLoadingVariantId] = useState<string | null>(null)

  const workerRef = useRef<Worker | null>(null)
  const readyRef = useRef(new Set<string>())
  const requestIdRef = useRef(0)
  const filesRef = useRef(new Map<string, { loaded: number; total: number }>())
  const pendingRef = useRef(
    new Map<
      number,
      {
        variantId: string
        resolve: (r: RunResult<any>) => void
        reject: (e: Error) => void
        onToken?: (text: string) => void
      }
    >(),
  )

  const rejectAllPending = useCallback((message: string) => {
    for (const { reject } of pendingRef.current.values()) reject(new Error(message))
    pendingRef.current.clear()
  }, [])

  const ensureWorker = useCallback((): Worker => {
    if (workerRef.current) return workerRef.current

    // `new URL(..., import.meta.url)` là cách Vite nhận ra đây là worker và
    // bundle nó thành file riêng. Không được ghép chuỗi đường dẫn.
    const worker = new Worker(new URL('../workers/pipeline.worker.ts', import.meta.url), {
      type: 'module',
    })

    worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data

      switch (msg.kind) {
        case 'progress': {
          if (msg.file && typeof msg.total === 'number' && msg.total > 0) {
            filesRef.current.set(msg.file, { loaded: msg.loaded ?? 0, total: msg.total })
            let loaded = 0
            let total = 0
            for (const f of filesRef.current.values()) {
              loaded += f.loaded
              total += f.total
            }
            setOverall(total > 0 ? Math.min(loaded / total, 1) : 0)
          }
          if (msg.file) setCurrentFile(msg.file)
          break
        }

        case 'loaded': {
          const variantId = msg.variantId ?? 'default'
          readyRef.current.add(variantId)
          setReadyVariants([...readyRef.current])
          setTimings((prev) => ({ ...prev, [variantId]: { ...prev[variantId], loadMs: msg.ms } }))
          setDevice(msg.device)
          setDeviceNote(msg.deviceNote ?? null)
          setLoadMs(msg.ms)
          setOverall(1)
          setCurrentFile(null)
          setLoadingVariantId(null)
          // Nếu người dùng bấm "Chạy" ngay (kéo theo cả load), giữ trạng thái
          // 'running' cho tới khi có kết quả.
          setStatus(pendingRef.current.size > 0 ? 'running' : 'ready')
          break
        }

        case 'token': {
          // Không đụng tới `status`: vẫn đang 'running' cho tới khi có 'result'.
          pendingRef.current.get(msg.requestId)?.onToken?.(msg.text)
          break
        }

        case 'result': {
          const pending = pendingRef.current.get(msg.requestId)
          pendingRef.current.delete(msg.requestId)
          const variantId = msg.variantId ?? 'default'
          setLastMs(msg.ms)
          setTimings((prev) => ({ ...prev, [variantId]: { ...prev[variantId], lastMs: msg.ms } }))
          setStatus(pendingRef.current.size > 0 ? 'running' : 'ready')
          pending?.resolve({ output: msg.output, debug: msg.debug, ms: msg.ms })
          break
        }

        case 'error': {
          setError(msg.message)
          setStatus('error')
          setLoadingVariantId(null)
          if (msg.variantId) {
            readyRef.current.delete(msg.variantId)
            setReadyVariants([...readyRef.current])
          }
          rejectAllPending(msg.message)
          break
        }
      }
    })

    worker.addEventListener('error', (event) => {
      const message = event.message || 'Worker gặp lỗi không xác định'
      setError(message)
      setStatus('error')
      rejectAllPending(message)
    })

    workerRef.current = worker
    return worker
  }, [rejectAllPending])

  /** Tải model trước, không chạy inference. Dùng cho nút "Tải model". */
  const load = useCallback(
    (variantId?: string) => {
      const worker = ensureWorker()
      const variant = resolveVariant(spec, variantId)
      setError(null)
      setStatus('loading')
      filesRef.current.clear()
      setOverall(0)
      setLoadingVariantId(variant.id)
      worker.postMessage({ kind: 'load', demoId, variantId: variant.id } satisfies WorkerRequest)
    },
    [demoId, ensureWorker, spec],
  )

  /** Chạy inference. Tự load model nếu chưa có. */
  const run = useCallback(
    <T = unknown>(input: unknown, runOptions?: RunOptions): Promise<RunResult<T>> => {
      const worker = ensureWorker()
      const variant = resolveVariant(spec, runOptions?.variantId)
      const requestId = ++requestIdRef.current
      const alreadyReady = readyRef.current.has(variant.id)

      setError(null)
      setStatus(alreadyReady ? 'running' : 'loading')
      if (!alreadyReady) {
        filesRef.current.clear()
        setOverall(0)
        setLoadingVariantId(variant.id)
      }

      return new Promise<RunResult<T>>((resolve, reject) => {
        pendingRef.current.set(requestId, {
          variantId: variant.id,
          resolve,
          reject,
          onToken: runOptions?.onToken,
        })
        worker.postMessage({
          kind: 'run',
          demoId,
          requestId,
          input,
          variantId: variant.id,
          options: runOptions?.pipelineOptions,
          args: runOptions?.args,
          // Hàm không qua được postMessage — chỉ gửi cờ, callback ở lại đây.
          stream: Boolean(runOptions?.onToken),
        } satisfies WorkerRequest)
      })
    },
    [demoId, ensureWorker, spec],
  )

  // Dọn worker khi card bị unmount, tránh giữ vài trăm MB trong RAM.
  useEffect(() => {
    return () => {
      workerRef.current?.terminate()
      workerRef.current = null
      readyRef.current.clear()
      pendingRef.current.clear()
    }
  }, [])

  const isVariantReady = useCallback(
    (variantId?: string) => readyVariants.includes(resolveVariant(spec, variantId).id),
    [readyVariants, spec],
  )

  return {
    spec,
    status,
    error,
    /** Backend thực tế sau khi resolve 'auto'. */
    device,
    /** Có nội dung khi scaffold tự đổi backend — xem resolveDevice trong worker. */
    deviceNote,
    loadMs,
    lastMs,
    /** 0–1 cho lần load đang diễn ra. */
    overall,
    currentFile,
    loadingVariantId,
    readyVariants,
    isVariantReady,
    timings,
    isBusy: status === 'loading' || status === 'running',
    isReady: status === 'ready' || status === 'running',
    load,
    run,
  }
}
