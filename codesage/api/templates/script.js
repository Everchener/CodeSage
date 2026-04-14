const { createApp } = Vue;

const CHAT_OVERALL_TIMEOUT_MS = 45000;
const CHAT_IDLE_TIMEOUT_MS = 20000;
const SESSION_STORAGE_KEY = "codesage_session_id";

function loadOrCreateSessionId() {
    try {
        const existing = localStorage.getItem(SESSION_STORAGE_KEY);
        if (existing) {
            return existing;
        }
        const created = `session_${Date.now()}`;
        localStorage.setItem(SESSION_STORAGE_KEY, created);
        return created;
    } catch (_error) {
        return `session_${Date.now()}`;
    }
}

createApp({
    data() {
        return {
            messages: [],
            userInput: "",
            isLoading: false,
            activeMode: "chat",
            API_URL: "/chat",
            abortController: null,
            abortReason: "",
            sessionId: loadOrCreateSessionId(),
            modifyWorkingDir: ".",
            modifyApprovalMode: "high_risk",
            isComposing: false,
            selectedFile: null,
            isUploading: false,
            uploadProgress: "",
            repoUrl: "",
            isIndexing: false,
            indexProgress: "",
            showHelp: false,
            activeBotMessageIndex: null,
            activeRunId: "",
            cancelledRunIds: {},
            recentRuns: [],
            recentRunsLoading: false,
            observerOpen: false,
            observerLoading: false,
            observerError: "",
            observerRun: null,
            observerEvents: [],
            observerToolCalls: [],
        };
    },
    computed: {
        statusText() {
            const statusMap = {
                chat: "CodeSage 在线",
                index: "知识库模式",
                review: "审查模式",
                modify: "修改模式",
            };
            return statusMap[this.activeMode] || "CodeSage 在线";
        },
        inputPlaceholder() {
            const placeholderMap = {
                chat: "输入需求，回车发送，Shift+Enter 换行",
                review: "输入 review 指令或 PR 链接",
                modify: "输入代码修改需求",
                index: "在下方管理仓库索引或文档上传",
            };
            return placeholderMap[this.activeMode] || "请输入内容";
        },
    },
    mounted() {
        this.configureMarked();
        try {
            localStorage.setItem(SESSION_STORAGE_KEY, this.sessionId);
        } catch (_error) {
            // Ignore storage errors and fall back to the in-memory session id.
        }
        this.fetchRecentRuns();
    },
    methods: {
        configureMarked() {
            marked.setOptions({
                highlight(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : "plaintext";
                    return hljs.highlight(code, { language }).value;
                },
                langPrefix: "hljs language-",
                breaks: true,
                gfm: true,
            });
        },

        parseMarkdown(text) {
            return marked.parse(text || "");
        },

        escapeHtml(text) {
            const div = document.createElement("div");
            div.textContent = text || "";
            return div.innerHTML;
        },

        switchMode(mode) {
            this.activeMode = mode;
            this.showHelp = false;
            if (mode === "chat") {
                this.API_URL = "/chat";
            } else if (mode === "review") {
                this.API_URL = "/review";
            } else if (mode === "modify") {
                this.API_URL = "/modify";
            }
        },

        handleCompositionStart() {
            this.isComposing = true;
        },

        handleCompositionEnd() {
            this.isComposing = false;
        },

        handleKeyDown(event) {
            if (event.key === "Enter" && !event.shiftKey && !this.isComposing) {
                event.preventDefault();
                this.handleSend();
            }
        },

        async handleStop() {
            if (this.abortController) {
                this.abortReason = "请求已取消。";
                if (this.activeMode === "chat" && this.activeRunId) {
                    this.markRunCancelled(this.activeRunId);
                    try {
                        await this.cancelChatRun(this.activeRunId);
                    } catch (error) {
                        console.warn("Chat cancellation request failed:", error);
                    }
                }
                this.abortController.abort();
            }
        },

        toggleCodeInput() {
            if (this.activeMode !== "chat") {
                this.activeMode = "chat";
                this.API_URL = "/chat";
            }
        },

        quickAction(text) {
            this.userInput = text;
            this.handleSend();
        },

        createPendingBotMessage() {
            return {
                text: "",
                isUser: false,
                state: "thinking",
                isThinking: true,
                thinkingText: this.getThinkingText(),
                timeline: [],
                error: null,
                threadId: "",
                runId: "",
                routeMeta: null,
                confirmation: null,
            };
        },

        getMessageAt(index) {
            return this.messages[index] || null;
        },

        formatRunStatus(status) {
            const value = String(status || "").trim();
            const labels = {
                running: "运行中",
                streaming: "输出中",
                awaiting_confirmation: "待确认",
                completed: "已完成",
                cancelled: "已取消",
                error: "失败",
                timed_out: "超时",
                unknown: "未知",
            };
            return labels[value] || value || "未知";
        },

        formatRunTime(value) {
            if (!value) {
                return "";
            }
            try {
                return new Date(Number(value) * 1000).toLocaleString("zh-CN", {
                    hour12: false,
                });
            } catch (_error) {
                return "";
            }
        },

        async fetchRecentRuns() {
            this.recentRunsLoading = true;
            try {
                const response = await fetch("/runs?limit=8");
                if (!response.ok) {
                    throw new Error(await this.parseErrorResponse(response));
                }
                const payload = await response.json();
                this.recentRuns = Array.isArray(payload.runs) ? payload.runs : [];
            } catch (error) {
                console.warn("Failed to load observed runs:", error);
            } finally {
                this.recentRunsLoading = false;
            }
        },

        async openObserver(runId = "") {
            const resolvedRunId = String(runId || this.activeRunId || "").trim();
            if (!resolvedRunId) {
                this.observerOpen = true;
                this.observerError = "当前还没有可查看的运行记录。";
                this.observerRun = null;
                this.observerEvents = [];
                this.observerToolCalls = [];
                return;
            }

            this.observerOpen = true;
            this.observerLoading = true;
            this.observerError = "";
            try {
                const response = await fetch(`/runs/${encodeURIComponent(resolvedRunId)}?limit=120`);
                if (!response.ok) {
                    throw new Error(await this.parseErrorResponse(response));
                }
                const payload = await response.json();
                this.observerRun = payload;
                this.observerEvents = Array.isArray(payload.events) ? payload.events : [];
                this.observerToolCalls = Array.isArray(payload.tool_calls) ? payload.tool_calls : [];
            } catch (error) {
                this.observerError = error.message;
                this.observerRun = null;
                this.observerEvents = [];
                this.observerToolCalls = [];
            } finally {
                this.observerLoading = false;
                this.fetchRecentRuns();
            }
        },

        closeObserver() {
            this.observerOpen = false;
        },

        openRelatedObserver(runId) {
            if (!runId) {
                return;
            }
            this.openObserver(runId);
        },

        finalizeBotMessage(index, state, options = {}) {
            const message = this.getMessageAt(index);
            if (!message) {
                return;
            }

            message.state = state;
            message.isThinking = state === "thinking" || state === "streaming";

            if (Object.prototype.hasOwnProperty.call(options, "text")) {
                message.text = options.text || "";
            }
            if (Object.prototype.hasOwnProperty.call(options, "thinkingText")) {
                message.thinkingText = options.thinkingText || "";
            }
            if (Object.prototype.hasOwnProperty.call(options, "routeMeta")) {
                message.routeMeta = options.routeMeta;
            }
            if (Object.prototype.hasOwnProperty.call(options, "confirmation")) {
                message.confirmation = options.confirmation;
            }
            if (options.runId) {
                message.runId = options.runId;
            }
            message.error = options.error || null;
            if (options.threadId) {
                message.threadId = options.threadId;
            }
        },

        markRunCancelled(runId) {
            if (!runId) {
                return;
            }
            this.cancelledRunIds[runId] = true;
        },

        isRunCancelled(runId) {
            return Boolean(runId && this.cancelledRunIds[runId]);
        },

        async cancelChatRun(runId) {
            const response = await fetch("/chat/cancel", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ run_id: runId }),
            });
            if (!response.ok) {
                throw new Error(await this.parseErrorResponse(response));
            }
            return response.json();
        },

        buildRequestPayload(text) {
            if (this.activeMode === "modify") {
                return {
                    instruction: text,
                    working_dir: this.modifyWorkingDir,
                    thread_id: this.sessionId,
                    approval_mode: this.modifyApprovalMode,
                };
            }
            return {
                message: text,
                thread_id: this.sessionId,
            };
        },

        buildRouteMeta(parsed) {
            if (!parsed.route && !parsed.mode && !parsed.agent) {
                return null;
            }
            return {
                route: parsed.route || "",
                mode: parsed.mode || "",
                agent: parsed.agent || "",
            };
        },

        buildConfirmationState(parsed, source = "chat") {
            return {
                source,
                status: "pending",
                previewId: parsed.preview_id || "",
                pendingChanges: parsed.pending_changes || [],
                riskReasons: parsed.risk_reasons || [],
                diffSummary: parsed.diff_summary || "",
                isSubmitting: false,
                error: null,
            };
        },

        formatTimelineEntry(parsed) {
            const labelMap = {
                accepted: "请求已受理",
                route: "路由决策",
                agent: "交给目标 Agent",
                self_rag: "判断是否检索",
                rewrite: "查询改写",
                retrieve: "检索知识库",
                auto_merge: "合并上下文",
                rerank: "结果重排",
                generate: "生成回答",
                analyze: "分析任务",
                plan: "生成计划",
                execute: "执行修改",
                execute_tool: "调用工具",
                verify: "验证结果",
                confirmation: "等待确认",
                output: "整理输出",
                complete: "执行完成",
                index_help: "索引说明",
                done: "完成",
            };

            const label = labelMap[parsed.stage] || parsed.stage || "处理中";
            const parts = [label];
            if (parsed.summary) {
                parts.push(parsed.summary);
            }
            if (parsed.tool) {
                parts.push(`工具: ${parsed.tool}`);
            }
            if (parsed.route || parsed.mode) {
                parts.push(`路由: ${parsed.route || "-"} / ${parsed.mode || "-"}`);
            }
            return {
                id: `${parsed.stage || "stage"}:${parsed.tool || ""}`,
                label,
                detail: parts.slice(1).join(" · "),
                status: parsed.status || "running",
                seq: Number(parsed.seq || 0),
            };
        },

        appendTimelineEntry(message, parsed) {
            if (!message || parsed.type !== "step") {
                return;
            }
            const entry = this.formatTimelineEntry(parsed);
            const existingIndex = message.timeline.findIndex((item) => item.id === entry.id);
            if (existingIndex >= 0) {
                const existing = message.timeline[existingIndex];
                message.timeline.splice(existingIndex, 1, {
                    ...existing,
                    detail: entry.detail || existing.detail,
                    status: entry.status || existing.status,
                    seq: Math.max(existing.seq || 0, entry.seq || 0),
                });
            } else {
                message.timeline.push(entry);
            }
            message.timeline.sort((a, b) => (a.seq || 0) - (b.seq || 0));
        },

        finalizeTimeline(message, finalState) {
            if (!message || !Array.isArray(message.timeline)) {
                return;
            }
            const status = finalState === "error" ? "error" : "completed";
            message.timeline = message.timeline.map((item) => ({
                ...item,
                status: item.status === "error" ? "error" : status,
            }));
        },

        formatModifyCompletionText(payload) {
            const changes = payload.applied_changes || payload.changes_made || [];
            const changesText = changes.length ? changes.map((item) => `- ${item}`).join("\n") : "- 无";
            const verification = payload.verification || payload.verification_result || "未返回验证结果。";
            return `代码修改已完成。\n\n已应用文件：\n${changesText}\n\n验证结果：\n${verification}`;
        },

        setAwaitingConfirmationMessage(botMsgIdx, payload, routeMeta) {
            const message = this.getMessageAt(botMsgIdx);
            if (!message) {
                return;
            }
            const runId = payload.run_id || message.runId || "";
            if (this.isRunCancelled(runId)) {
                message.confirmation = null;
                return;
            }

            this.appendTimelineEntry(message, {
                type: "step",
                stage: "confirmation",
                status: "running",
                summary: "高风险修改预览已生成，等待用户确认。",
                route: routeMeta?.route || "modify",
                mode: routeMeta?.mode || "direct",
                agent: routeMeta?.agent || "code_modify_agent",
            });
            this.finalizeBotMessage(botMsgIdx, "awaiting_confirmation", {
                text: payload.content || "已生成代码修改预览，等待确认。",
                threadId: payload.thread_id || message.threadId || this.sessionId,
                runId,
                routeMeta,
                confirmation: this.buildConfirmationState(payload, routeMeta?.mode === "direct" ? "modify" : "chat"),
            });
        },

        async handleSend() {
            const text = this.userInput.trim();
            if (!text || this.isLoading || this.isComposing || this.activeMode === "index") {
                return;
            }

            this.messages.push({ text, isUser: true });
            this.userInput = "";
            this.$nextTick(() => {
                this.resetTextareaHeight();
                this.scrollToBottom();
            });

            this.isLoading = true;
            this.messages.push(this.createPendingBotMessage());
            const botMsgIdx = this.messages.length - 1;
            this.activeBotMessageIndex = botMsgIdx;
            this.activeRunId = "";
            this.abortController = new AbortController();
            this.abortReason = "";

            try {
                await this.handleChatRequest(text, botMsgIdx);
            } catch (error) {
                const aborted = error.name === "AbortError" || this.abortReason;
                const botMessage = this.getMessageAt(botMsgIdx);
                if (aborted) {
                    this.finalizeBotMessage(botMsgIdx, "cancelled", {
                        text: this.abortReason || "请求已取消。",
                        error: this.abortReason || null,
                    });
                } else {
                    console.error("Error:", error);
                    if (!botMessage || !["done", "error", "cancelled", "awaiting_confirmation"].includes(botMessage.state)) {
                        this.finalizeBotMessage(botMsgIdx, "error", {
                            text: `请求失败：${error.message}`,
                            error: error.message,
                        });
                    }
                }
            } finally {
                this.isLoading = false;
                this.abortController = null;
                this.abortReason = "";
                this.activeBotMessageIndex = null;
                this.activeRunId = "";
                this.fetchRecentRuns();
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        async handleJsonResponse(payload, botMsgIdx) {
            const routeMeta =
                this.activeMode === "modify"
                    ? { route: "modify", mode: "direct", agent: "code_modify_agent" }
                    : this.activeMode === "review"
                        ? { route: "review", mode: "direct", agent: "pr_review_agent" }
                        : { route: this.activeMode, mode: "direct", agent: this.activeMode };

            const message = this.getMessageAt(botMsgIdx);
            if (!message) {
                return;
            }

            if (this.activeMode === "modify") {
                if (payload.status === "awaiting_confirmation") {
                    this.setAwaitingConfirmationMessage(botMsgIdx, payload, routeMeta);
                    return;
                }
                if (payload.status === "completed") {
                    this.appendTimelineEntry(message, {
                        type: "step",
                        stage: "done",
                        status: "completed",
                        summary: "代码修改已完成。",
                        route: routeMeta.route,
                        mode: routeMeta.mode,
                        agent: routeMeta.agent,
                    });
                    this.finalizeTimeline(message, "done");
                    this.finalizeBotMessage(botMsgIdx, "done", {
                        text: this.formatModifyCompletionText(payload),
                        routeMeta,
                        threadId: this.sessionId,
                        runId: payload.run_id || "",
                        confirmation: null,
                    });
                    return;
                }
                throw new Error(payload.error || "代码修改失败。");
            }

            if (this.activeMode === "review") {
                const text = payload.final_comment || payload.message || JSON.stringify(payload, null, 2);
                this.finalizeBotMessage(botMsgIdx, "done", {
                    text,
                    routeMeta,
                    threadId: this.sessionId,
                    runId: payload.run_id || "",
                });
                return;
            }

            this.finalizeBotMessage(botMsgIdx, "done", {
                text: JSON.stringify(payload, null, 2),
                routeMeta,
                threadId: this.sessionId,
                runId: payload.run_id || "",
            });
        },

        async handlePreviewDecision(messageIndex, decision) {
            const message = this.getMessageAt(messageIndex);
            if (!message || !message.confirmation || message.confirmation.isSubmitting) {
                return;
            }

            message.confirmation.isSubmitting = true;
            message.confirmation.error = null;
            try {
                const response = await fetch("/modify/confirm", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        preview_id: message.confirmation.previewId,
                        decision,
                        thread_id: message.threadId || this.sessionId,
                    }),
                });
                if (!response.ok) {
                    throw new Error(await this.parseErrorResponse(response));
                }

                const payload = await response.json();
                if (payload.status === "cancelled") {
                    this.appendTimelineEntry(message, {
                        type: "step",
                        stage: "confirmation",
                        status: "completed",
                        summary: "用户已取消本次修改。",
                        route: message.routeMeta?.route || "modify",
                        mode: message.routeMeta?.mode || "direct",
                        agent: message.routeMeta?.agent || "code_modify_agent",
                    });
                    this.finalizeBotMessage(messageIndex, "cancelled", {
                        text: "已取消本次代码修改，工作区未落盘。",
                        routeMeta: message.routeMeta,
                        confirmation: null,
                    });
                    return;
                }

                this.appendTimelineEntry(message, {
                    type: "step",
                    stage: "confirmation",
                    status: "completed",
                    summary: "预览已确认并应用到工作区。",
                    route: message.routeMeta?.route || "modify",
                    mode: message.routeMeta?.mode || "direct",
                    agent: message.routeMeta?.agent || "code_modify_agent",
                });
                this.finalizeTimeline(message, "done");
                this.finalizeBotMessage(messageIndex, "done", {
                    text: this.formatModifyCompletionText(payload),
                    routeMeta: message.routeMeta,
                    runId: payload.run_id || message.runId || "",
                    confirmation: null,
                });
            } catch (error) {
                console.error("Preview confirmation failed:", error);
                message.confirmation.error = error.message;
            } finally {
                if (message.confirmation) {
                    message.confirmation.isSubmitting = false;
                }
                this.fetchRecentRuns();
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        async handleChatRequest(text, botMsgIdx) {
            let overallTimer = null;
            let idleTimer = null;

            const clearTimers = () => {
                if (overallTimer) {
                    clearTimeout(overallTimer);
                    overallTimer = null;
                }
                if (idleTimer) {
                    clearTimeout(idleTimer);
                    idleTimer = null;
                }
            };

            const abortWithReason = (reason) => {
                this.abortReason = reason;
                if (this.abortController) {
                    this.abortController.abort();
                }
            };

            const resetIdleTimer = () => {
                if (idleTimer) {
                    clearTimeout(idleTimer);
                }
                idleTimer = setTimeout(() => {
                    abortWithReason("等待模型或工具链响应超时。");
                }, CHAT_IDLE_TIMEOUT_MS);
            };

            let terminalEventSeen = false;
            let messageEventSeen = false;

            const processEventBlock = (eventBlock) => {
                const lines = eventBlock.split("\n");
                let eventName = "";
                const dataLines = [];

                for (const line of lines) {
                    const trimmedLine = line.trim();
                    if (!trimmedLine) {
                        continue;
                    }
                    if (trimmedLine.startsWith("event:")) {
                        eventName = trimmedLine.slice(6).trim();
                    } else if (trimmedLine.startsWith("data:")) {
                        dataLines.push(trimmedLine.slice(5).trim());
                    }
                }

                if (!dataLines.length) {
                    return;
                }

                let parsed;
                try {
                    parsed = JSON.parse(dataLines.join("\n"));
                } catch (error) {
                    throw new Error(`聊天流数据格式无效：${error.message}`);
                }

                const eventType = parsed.type || eventName;
                const message = this.getMessageAt(botMsgIdx);
                if (!message || (terminalEventSeen && eventType !== "confirmation_required")) {
                    return;
                }
                const parsedRunId = typeof parsed.run_id === "string" ? parsed.run_id : "";
                if (parsedRunId) {
                    message.runId = parsedRunId;
                    if (this.activeBotMessageIndex === botMsgIdx) {
                        this.activeRunId = parsedRunId;
                    }
                }
                const effectiveRunId = parsedRunId || message.runId || "";
                if (this.isRunCancelled(effectiveRunId)) {
                    return;
                }

                const routeMeta = this.buildRouteMeta(parsed) || message.routeMeta;

                if (eventType === "step") {
                    this.finalizeBotMessage(botMsgIdx, "thinking", {
                        text: message.text,
                        thinkingText: parsed.summary || parsed.stage || "处理中...",
                        threadId: parsed.thread_id || message.threadId,
                        runId: effectiveRunId,
                        routeMeta,
                    });
                    this.appendTimelineEntry(message, parsed);
                } else if (eventType === "message") {
                    messageEventSeen = true;
                    this.finalizeBotMessage(botMsgIdx, "streaming", {
                        text: parsed.content || "",
                        threadId: parsed.thread_id || message.threadId,
                        runId: effectiveRunId,
                        routeMeta,
                    });
                } else if (eventType === "confirmation_required") {
                    this.setAwaitingConfirmationMessage(botMsgIdx, parsed, routeMeta);
                } else if (eventType === "error") {
                    terminalEventSeen = true;
                    const finalState = parsed.status === "cancelled" ? "cancelled" : "error";
                    this.appendTimelineEntry(message, {
                        ...parsed,
                        type: "step",
                        stage: parsed.stage || "error",
                        status: finalState === "cancelled" ? "completed" : "error",
                    });
                    this.finalizeTimeline(message, finalState === "error" ? "error" : "done");
                    this.finalizeBotMessage(botMsgIdx, finalState, {
                        text: parsed.content || (finalState === "cancelled" ? "请求已取消。" : "对话请求失败。"),
                        error: finalState === "error" ? (parsed.content || "对话请求失败。") : null,
                        threadId: parsed.thread_id || message.threadId,
                        runId: effectiveRunId,
                        routeMeta,
                    });
                } else if (eventType === "done") {
                    terminalEventSeen = true;
                    const finalText = parsed.content || message.text || "请求已完成。";
                    const finalState =
                        parsed.status === "awaiting_confirmation"
                            ? "awaiting_confirmation"
                            : parsed.status === "cancelled"
                                ? "cancelled"
                            : parsed.status && !["success", "completed"].includes(parsed.status)
                                ? "error"
                                : "done";

                    this.appendTimelineEntry(message, {
                        ...parsed,
                        type: "step",
                        stage: "done",
                        status: finalState === "error" ? "error" : "completed",
                    });
                    if (finalState !== "awaiting_confirmation") {
                        this.finalizeTimeline(message, finalState === "error" ? "error" : "done");
                    }
                    this.finalizeBotMessage(botMsgIdx, finalState, {
                        text: finalText,
                        error: finalState === "error" ? finalText : null,
                        threadId: parsed.thread_id || message.threadId,
                        runId: effectiveRunId,
                        routeMeta,
                    });
                }

                this.$nextTick(() => this.scrollToBottom());
            };

            overallTimer = setTimeout(() => {
                abortWithReason("等待模型或工具链响应超时。");
            }, CHAT_OVERALL_TIMEOUT_MS);

            try {
                const response = await fetch(this.API_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(this.buildRequestPayload(text)),
                    signal: this.abortController.signal,
                });

                if (!response.ok) {
                    throw new Error(await this.parseErrorResponse(response));
                }

                const contentType = response.headers.get("content-type") || "";
                if (!contentType.includes("text/event-stream")) {
                    const payload = await response.json().catch(() => ({}));
                    await this.handleJsonResponse(payload, botMsgIdx);
                    return;
                }

                if (!response.body || typeof response.body.getReader !== "function") {
                    throw new Error("Streaming response is not available.");
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = "";

                resetIdleTimer();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        break;
                    }

                    resetIdleTimer();
                    buffer += decoder.decode(value, { stream: true });
                    const events = buffer.split("\n\n");
                    buffer = events.pop() || "";

                    for (const eventBlock of events) {
                        processEventBlock(eventBlock);
                    }
                }

                buffer += decoder.decode();
                if (buffer.trim()) {
                    processEventBlock(buffer.trim());
                }

                const message = this.getMessageAt(botMsgIdx);
                if (message && !terminalEventSeen) {
                    if (messageEventSeen || message.text) {
                        this.finalizeBotMessage(botMsgIdx, "done", {
                            text: message.text || "请求已完成。",
                        });
                    } else {
                        this.finalizeBotMessage(botMsgIdx, "error", {
                            text: "对话在完成前连接已关闭。",
                            error: "对话在完成前连接已关闭。",
                        });
                    }
                }
            } finally {
                clearTimers();
            }
        },

        async parseErrorResponse(response) {
            const payload = await response.json().catch(() => ({}));
            if (typeof payload.detail === "string") {
                return payload.detail;
            }
            if (typeof payload.message === "string") {
                return payload.message;
            }
            return `HTTP ${response.status}`;
        },

        getThinkingText() {
            return this.activeMode === "index" ? "处理中..." : "正在协同多个 Agent...";
        },

        autoResize(event) {
            const textarea = event.target;
            textarea.style.height = "auto";
            textarea.style.height = `${textarea.scrollHeight}px`;
        },

        resetTextareaHeight() {
            if (this.$refs.textarea) {
                this.$refs.textarea.style.height = "auto";
            }
        },

        scrollToBottom() {
            if (this.$refs.chatContainer) {
                this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
            }
        },

        handleClearChat() {
            if (confirm("确认清空当前会话吗？")) {
                this.messages = [];
                this.sessionId = `session_${Date.now()}`;
                this.activeBotMessageIndex = null;
                this.activeRunId = "";
                this.cancelledRunIds = {};
                try {
                    localStorage.setItem(SESSION_STORAGE_KEY, this.sessionId);
                } catch (_error) {
                    // 忽略存储异常。
                }
            }
        },

        handleFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                this.selectedFile = files[0];
                this.uploadProgress = "";
            }
        },

        async uploadDocument() {
            if (!this.selectedFile) {
                alert("请先选择文件。");
                return;
            }

            this.isUploading = true;
            this.uploadProgress = "文档上传中...";

            try {
                const formData = new FormData();
                formData.append("file", this.selectedFile);

                const response = await fetch("/index_docs", {
                    method: "POST",
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error(await this.parseErrorResponse(response));
                }

                const data = await response.json();
                this.uploadProgress = data.message || "文档上传成功。";
                this.selectedFile = null;
                if (this.$refs.fileInput) {
                    this.$refs.fileInput.value = "";
                }
            } catch (error) {
                console.error("Error uploading document:", error);
                this.uploadProgress = `上传失败：${error.message}`;
            } finally {
                this.isUploading = false;
            }
        },

        async indexRepo() {
            if (!this.repoUrl.trim()) {
                alert("请输入本地仓库路径。");
                return;
            }

            this.isIndexing = true;
            this.indexProgress = "正在启动仓库索引...";

            try {
                const response = await fetch("/index", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ repo_path: this.repoUrl.trim() }),
                });

                if (!response.ok) {
                    throw new Error(await this.parseErrorResponse(response));
                }

                const data = await response.json();
                this.indexProgress = data.status === "indexing started"
                    ? `已开始索引：${data.repo_path}`
                    : JSON.stringify(data);
            } catch (error) {
                console.error("Error indexing repo:", error);
                this.indexProgress = `索引失败：${error.message}`;
            } finally {
                this.isIndexing = false;
            }
        },
    },
    watch: {
        messages: {
            handler() {
                this.$nextTick(() => this.scrollToBottom());
            },
            deep: true,
        },
    },
}).mount("#app");
