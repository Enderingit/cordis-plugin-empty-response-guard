import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { FinishReason, GenerateOptions, LlmFailure, StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm'
import { EMPTY_RESPONSE_CODE, chunkHasVisibleText } from '@deepseek-ai/dsh-llm'

export const name = 'empty-response-guard'
export const inject = ['llm']
export const TRANSPORT_CODE = 'TRANSPORT'

export interface Config {
  mode?: 'error' | 'warn' | 'off'
  minReasoningChars?: number
  reportReasoning?: boolean
  detectFabricatedStop?: boolean
  fabricatedStopNeedsCalibration?: boolean
  fabricatedStopMaxPerSession?: number
}
export type ResolvedConfig = Required<Config>
export const Config: z<Config> = z.object({
  mode: z.union(['error', 'warn', 'off']).default('error'),
  minReasoningChars: z.number().min(0).default(0),
  reportReasoning: z.boolean().default(true),
  detectFabricatedStop: z.boolean().default(true),
  fabricatedStopNeedsCalibration: z.boolean().default(true),
  fabricatedStopMaxPerSession: z.number().min(0).default(3),
})
export const DEFAULT_MODE = 'error'
export const DEFAULT_FABRICATED_STOP_MAX_PER_SESSION = 3

export function emptyResponseFailure(reasoningChars: number): LlmFailure {
  return { code: EMPTY_RESPONSE_CODE, message: `model returned a completed response with no text or tool call (${reasoningChars} characters of reasoning only)` }
}
export function emptyResponseFinish(reasoningChars: number): FinishReason {
  return { kind: 'error', failure: emptyResponseFailure(reasoningChars) }
}
export function usageWasReported(usage: TokenUsage): boolean {
  return usage.inputTokens !== 0 || usage.outputTokens !== 0 || (usage.totalTokens ?? 0) !== 0
    || (usage.cacheReadTokens ?? 0) !== 0 || (usage.cacheWriteTokens ?? 0) !== 0 || (usage.reasoningTokens ?? 0) !== 0
}
export function fabricatedStopFailure(detail: string): LlmFailure {
  return { code: TRANSPORT_CODE, message: `model reported a completed response whose usage accounts for no tokens at all (${detail})` }
}
export function fabricatedStopFinish(detail: string): FinishReason {
  return { kind: 'error', failure: fabricatedStopFailure(detail) }
}
export function isVisibleChunk(chunk: StreamChunk): boolean { return chunkHasVisibleText(chunk) }
export function degenerateReasoningChars(chunks: readonly StreamChunk[]): number | undefined {
  const finish = chunks.at(-1)
  if (finish?.type !== 'finish' || finish.reason.kind !== 'stop' || chunks.some(isVisibleChunk)) return undefined
  return chunks.reduce((sum, chunk) => sum + (chunk.type === 'reasoning-delta' ? chunk.text.length : 0), 0)
}
export function routeKeyOf(options: Pick<GenerateOptions, 'provider' | 'model'>): string {
  return `${options.provider}::${options.model}`
}
export interface GuardState { calibratedRoutes: Set<string>; corrections: Map<string, number> }
export function newGuardState(): GuardState { return { calibratedRoutes: new Set(), corrections: new Map() } }
export type SuppressionReason = 'cap' | 'uncalibrated' | 'warn-mode'
export interface CorrectionReport {
  kind: 'empty-response' | 'fabricated-stop'
  code: string
  corrected: boolean
  suppressedBy?: SuppressionReason
  reasoningChars: number
  visibleChars: number
  sessionId?: string
  route?: string
}
export interface GuardContext {
  state?: GuardState
  route?: string
  sessionId?: string
  onCorrection?: (report: CorrectionReport) => void
  onCalibrated?: (route: string) => void
}

export async function* guardStream(source: AsyncIterable<StreamChunk>, config: ResolvedConfig, context: GuardContext = {}): AsyncIterable<StreamChunk> {
  if (config.mode === 'off') { yield* source; return }
  let sawVisible = false
  let reasoningChars = 0
  let visibleChars = 0
  let usage: TokenUsage | undefined
  for await (const chunk of source) {
    if (chunk.type === 'usage') {
      usage = chunk.usage
      if (context.route !== undefined && usageWasReported(usage) && context.state?.calibratedRoutes.has(context.route) === false) {
        context.state.calibratedRoutes.add(context.route)
        context.onCalibrated?.(context.route)
      }
    } else if (chunk.type === 'reasoning-delta') reasoningChars += chunk.text.length
    else if (chunk.type === 'text-delta') visibleChars += chunk.text.length
    if (chunk.type === 'finish' && chunk.reason.kind === 'stop') {
      const base = {
        reasoningChars, visibleChars,
        ...(context.sessionId === undefined ? {} : { sessionId: context.sessionId }),
        ...(context.route === undefined ? {} : { route: context.route }),
      }
      let report: CorrectionReport | undefined
      let finish: FinishReason | undefined
      if (!sawVisible && reasoningChars >= config.minReasoningChars) {
        const corrected = config.mode === 'error'
        report = { ...base, kind: 'empty-response', code: EMPTY_RESPONSE_CODE, corrected, ...(!corrected ? { suppressedBy: 'warn-mode' as const } : {}) }
        if (corrected) finish = emptyResponseFinish(reasoningChars)
      } else if (config.detectFabricatedStop && usage !== undefined && !usageWasReported(usage)) {
        const detail = `${visibleChars} characters of text, ${reasoningChars} characters of reasoning, usage 0 tokens`
        report = { ...base, kind: 'fabricated-stop', code: TRANSPORT_CODE, corrected: false }
        if (config.mode === 'warn') report = { ...report, suppressedBy: 'warn-mode' }
        else if (config.fabricatedStopNeedsCalibration && context.route !== undefined && context.state?.calibratedRoutes.has(context.route) !== true) report = { ...report, suppressedBy: 'uncalibrated' }
        else {
          const key = context.sessionId ?? '*'
          const count = context.state?.corrections.get(key) ?? 0
          if (config.fabricatedStopMaxPerSession > 0 && count >= config.fabricatedStopMaxPerSession) report = { ...report, suppressedBy: 'cap' }
          else {
            context.state?.corrections.set(key, count + 1)
            finish = fabricatedStopFinish(detail)
            report = { ...report, corrected: true }
          }
        }
      }
      if (report) {
        context.onCorrection?.(report)
        if (finish) { yield { ...chunk, reason: finish }; continue }
      }
      yield chunk
      continue
    }
    if (!sawVisible && isVisibleChunk(chunk)) sawVisible = true
    yield chunk
  }
}

export async function* guardEmptyResponseStream(source: AsyncIterable<StreamChunk>, config: ResolvedConfig, onDegenerate?: (reasoningChars: number) => void): AsyncIterable<StreamChunk> {
  yield* guardStream(source, config, { state: newGuardState(), ...(onDegenerate ? { onCorrection: report => onDegenerate(report.reasoningChars) } : {}) })
}

export function apply(ctx: Context, config: ResolvedConfig): void {
  const resolved: ResolvedConfig = {
    ...config,
    mode: config.mode ?? DEFAULT_MODE,
    minReasoningChars: config.minReasoningChars ?? 0,
    reportReasoning: config.reportReasoning ?? true,
    detectFabricatedStop: config.detectFabricatedStop ?? true,
    fabricatedStopNeedsCalibration: config.fabricatedStopNeedsCalibration ?? true,
    fabricatedStopMaxPerSession: config.fabricatedStopMaxPerSession ?? DEFAULT_FABRICATED_STOP_MAX_PER_SESSION,
  }
  const state = newGuardState()
  ctx.on('llm/stream', (options: GenerateOptions, next: () => AsyncIterable<StreamChunk>) => {
    const upstream = next()
    if (resolved.mode === 'off') return upstream
    const route = routeKeyOf(options)
    return guardStream(upstream, resolved, {
      state, route,
      ...(options.sessionId === undefined ? {} : { sessionId: String(options.sessionId) }),
      onCalibrated: key => ctx.logger.info('empty-response-guard: route "%s" reports usage; a stop whose usage accounts for no tokens will now be treated as a fabricated completion (discussion #6948)', key),
      onCorrection: report => log(ctx, resolved, report),
    })
  })
}
function log(ctx: Context, config: ResolvedConfig, report: CorrectionReport): void {
  if (report.kind === 'empty-response') {
    ctx.logger.warn('empty-response-guard: model returned a completed response with no text or tool call (%d characters of reasoning only); %s (discussion #6218)', report.reasoningChars, report.corrected ? `reported as ${report.code} so the retry policy can act` : 'left as a successful stop (warn mode)')
    return
  }
  const observed = `route "${report.route ?? 'unknown'}" ${report.visibleChars} characters of text, ${report.reasoningChars} characters of reasoning`
  const outcome = report.corrected ? `reported as ${report.code} so the retry policy can act`
    : report.suppressedBy === 'cap' ? `left as a successful stop, the per-session bound is reached (fabricatedStopMaxPerSession=${config.fabricatedStopMaxPerSession})`
      : report.suppressedBy === 'uncalibrated' ? 'left as a successful stop, this route has not reported usage in this process (no calibration)' : 'left as a successful stop (warn mode)'
  ctx.logger.warn('empty-response-guard: a model response on %s ended with a `stop` whose usage accounts for no tokens at all; %s (discussion #6948)', observed, outcome)
}
