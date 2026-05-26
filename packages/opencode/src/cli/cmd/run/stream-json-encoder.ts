/**
 * Claude Code-compatible stream-json encoder for `opencode run`.
 *
 * Converts OpenCode's internal SSE events into the same NDJSON frame format
 * that Claude Code emits with `--output-format=stream-json`. This lets CAP
 * (CLI Agent Protocol) drive OpenCode with the same `ClaudeCodeDriver`
 * parser used for Claude Code and OpenClaude.
 *
 * Frame types emitted:
 *   - system/init     — session start (session ID + model)
 *   - assistant       — completed content blocks (text, thinking, tool_use)
 *   - user            — tool results (tool_result)
 *   - stream_event    — token-level deltas (content_block_delta)
 *   - result          — session end (usage stats + stop reason)
 *
 * @see https://github.com/rsclaw-ai/cap-protocol — CAP consumer
 */

import { EOL } from "os"

// ---------------------------------------------------------------------------
// Claude Code-compatible frame types
// ---------------------------------------------------------------------------

interface SystemInitFrame {
  type: "system"
  subtype: "init"
  session_id: string
  model?: string
}

interface AssistantFrame {
  type: "assistant"
  message: {
    id: string
    content: ContentBlock[]
  }
}

interface UserFrame {
  type: "user"
  message: {
    content: ToolResultBlock[]
  }
}

interface ResultFrame {
  type: "result"
  subtype: string
  duration_ms: number
  total_cost_usd: number
  usage: {
    input_tokens: number
    output_tokens: number
    cache_read_input_tokens: number
    cache_creation_input_tokens: number
  }
}

interface StreamEventFrame {
  type: "stream_event"
  event: {
    type: string
    index: number
    delta: { type: string; text?: string; thinking?: string }
  }
}

// ---------------------------------------------------------------------------
// Content block types (Claude API format)
// ---------------------------------------------------------------------------

type ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock

interface TextBlock {
  type: "text"
  text: string
}

interface ThinkingBlock {
  type: "thinking"
  thinking: string
}

interface ToolUseBlock {
  type: "tool_use"
  id: string
  name: string
  input: Record<string, unknown>
}

interface ToolResultBlock {
  type: "tool_result"
  tool_use_id: string
  content: string
  is_error?: boolean
}

// ---------------------------------------------------------------------------
// OpenCode SSE event shapes (minimal subset we consume)
// ---------------------------------------------------------------------------

interface SseEvent {
  type: string
  properties: Record<string, unknown>
}

interface ToolPartData {
  type: "tool"
  id: string
  sessionID: string
  messageID: string
  callID: string
  tool: string
  state:
    | { status: "pending"; input: Record<string, unknown>; raw: string }
    | {
        status: "running"
        input: Record<string, unknown>
        title?: string
        time: { start: number }
      }
    | {
        status: "completed"
        input: Record<string, unknown>
        output: string
        title: string
        time: { start: number; end: number }
      }
    | {
        status: "error"
        input: Record<string, unknown>
        error: string
        time: { start: number; end: number }
      }
}

interface TextPartData {
  type: "text"
  id: string
  sessionID: string
  messageID: string
  text: string
  time: { start: number; end?: number }
}

interface ReasoningPartData {
  type: "reasoning"
  id: string
  sessionID: string
  messageID: string
  text: string
  time: { start: number; end?: number }
}

interface StepFinishPartData {
  type: "step-finish"
  id: string
  sessionID: string
  cost: number
  tokens: {
    input: number
    output: number
    reasoning: number
    cache: { read: number; write: number }
  }
}

type PartData = ToolPartData | TextPartData | ReasoningPartData | StepFinishPartData | { type: string; [k: string]: unknown }

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/**
 * Stateful encoder that converts OpenCode SSE events to Claude Code
 * stream-json frames. One encoder instance per session.
 */
export class StreamJsonEncoder {
  private sessionID = ""
  private startTime = 0
  private model = ""
  private initEmitted = false

  // Part tracking: partID → type
  private partTypes = new Map<string, string>()

  // De-dup: which part IDs have been emitted as assistant frames
  private emittedParts = new Set<string>()

  // Accumulated usage from step-finish parts
  private usage = {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  }
  private totalCost = 0

  /**
   * Call once when the session starts, before processing events.
   * Records the session ID and start time for the result frame.
   */
  init(sessionID: string): void {
    this.sessionID = sessionID
    this.startTime = Date.now()
  }

  /**
   * Process one SSE event. Emits zero or more stream-json frames to stdout.
   * Returns `"break"` when the caller should stop consuming events
   * (session idle or terminal error).
   */
  processEvent(event: SseEvent): "break" | void {
    switch (event.type) {
      case "message.updated":
        return this.handleMessageUpdated(event.properties)

      case "message.part.updated":
        return this.handlePartUpdated(event.properties)

      case "message.part.delta":
        return this.handlePartDelta(event.properties)

      case "session.status": {
        const props = event.properties
        if (props.sessionID !== this.sessionID) return
        if ((props.status as { type?: string })?.type === "idle") {
          this.emitResult("success")
          return "break"
        }
        return
      }

      case "session.error": {
        const props = event.properties
        if (props.sessionID !== this.sessionID) return
        this.emitResult("error_during_execution")
        return "break"
      }
    }
  }

  // -----------------------------------------------------------------------
  // Event handlers
  // -----------------------------------------------------------------------

  private handleMessageUpdated(props: Record<string, unknown>): void {
    if (props.sessionID !== this.sessionID) return
    const info = props.info as { role?: string; modelID?: string } | undefined
    if (!info || info.role !== "assistant") return

    // Emit init frame on first assistant message (model becomes known here)
    if (!this.initEmitted && info.modelID) {
      this.model = info.modelID
      this.emitInit()
    }
  }

  private handlePartUpdated(props: Record<string, unknown>): void {
    const part = props.part as PartData | undefined
    if (!part || part.sessionID !== this.sessionID) return

    // Track part type for delta lookup
    this.partTypes.set(part.id, part.type)

    switch (part.type) {
      case "tool":
        this.handleToolPart(part as ToolPartData)
        break
      case "text":
        this.handleTextPart(part as TextPartData)
        break
      case "reasoning":
        this.handleReasoningPart(part as ReasoningPartData)
        break
      case "step-finish":
        this.handleStepFinish(part as StepFinishPartData)
        break
    }
  }

  private handlePartDelta(props: Record<string, unknown>): void {
    if (props.sessionID !== this.sessionID) return

    const partID = props.partID as string
    const partType = this.partTypes.get(partID)
    const delta = props.delta as string

    if (partType === "text" && props.field === "text") {
      this.writeFrame({
        type: "stream_event",
        event: {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: delta },
        },
      } satisfies StreamEventFrame)
    } else if (partType === "reasoning" && props.field === "text") {
      this.writeFrame({
        type: "stream_event",
        event: {
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: delta },
        },
      } satisfies StreamEventFrame)
    }
  }

  // -----------------------------------------------------------------------
  // Part handlers
  // -----------------------------------------------------------------------

  private handleToolPart(part: ToolPartData): void {
    const state = part.state

    // Emit tool_use as assistant frame once (on transition to running)
    if (
      (state.status === "running" || state.status === "completed") &&
      !this.emittedParts.has(part.id)
    ) {
      this.emittedParts.add(part.id)
      this.writeFrame({
        type: "assistant",
        message: {
          id: part.messageID,
          content: [
            {
              type: "tool_use",
              id: part.callID,
              name: part.tool,
              input: state.input,
            },
          ],
        },
      } satisfies AssistantFrame)
    }

    // Emit tool_result as user frame on completion/error
    if (state.status === "completed" && !this.emittedParts.has("result:" + part.id)) {
      this.emittedParts.add("result:" + part.id)
      this.writeFrame({
        type: "user",
        message: {
          content: [
            {
              type: "tool_result",
              tool_use_id: part.callID,
              content: state.output,
            },
          ],
        },
      } satisfies UserFrame)
    } else if (state.status === "error" && !this.emittedParts.has("result:" + part.id)) {
      this.emittedParts.add("result:" + part.id)
      this.writeFrame({
        type: "user",
        message: {
          content: [
            {
              type: "tool_result",
              tool_use_id: part.callID,
              content: state.error,
              is_error: true,
            },
          ],
        },
      } satisfies UserFrame)
    }
  }

  private handleTextPart(part: TextPartData): void {
    // Only emit on completion (time.end set); skip empty text
    if (!part.time?.end || this.emittedParts.has(part.id)) return
    this.emittedParts.add(part.id)

    const text = part.text
    if (!text) return

    this.writeFrame({
      type: "assistant",
      message: {
        id: part.messageID,
        content: [{ type: "text", text }],
      },
    } satisfies AssistantFrame)
  }

  private handleReasoningPart(part: ReasoningPartData): void {
    if (!part.time?.end || this.emittedParts.has(part.id)) return
    this.emittedParts.add(part.id)

    const text = part.text
    if (!text) return

    this.writeFrame({
      type: "assistant",
      message: {
        id: part.messageID,
        content: [{ type: "thinking", thinking: text }],
      },
    } satisfies AssistantFrame)
  }

  private handleStepFinish(part: StepFinishPartData): void {
    this.usage.input_tokens += part.tokens.input
    this.usage.output_tokens += part.tokens.output
    this.usage.cache_read_input_tokens += part.tokens.cache.read
    this.usage.cache_creation_input_tokens += part.tokens.cache.write
    this.totalCost += part.cost
  }

  // -----------------------------------------------------------------------
  // Frame emitters
  // -----------------------------------------------------------------------

  private emitInit(): void {
    this.initEmitted = true
    this.writeFrame({
      type: "system",
      subtype: "init",
      session_id: this.sessionID,
      model: this.model,
    } satisfies SystemInitFrame)
  }

  private resultEmitted = false

  private emitResult(subtype: string): void {
    if (this.resultEmitted) return
    this.resultEmitted = true
    // Ensure init was emitted even if no assistant message arrived
    if (!this.initEmitted) {
      this.emitInit()
    }
    this.writeFrame({
      type: "result",
      subtype,
      duration_ms: Date.now() - this.startTime,
      total_cost_usd: this.totalCost,
      usage: { ...this.usage },
    } satisfies ResultFrame)
  }

  /**
   * Ensure result frame is emitted. Call this when the event stream ends
   * (e.g., stream closes before session.status idle arrives).
   */
  finalize(): void {
    if (!this.resultEmitted) {
      this.emitResult("success")
    }
  }

  private writeFrame(frame: object): void {
    process.stdout.write(JSON.stringify(frame) + EOL)
  }
}
