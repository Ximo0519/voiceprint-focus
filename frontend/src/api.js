export function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws`;
}

export class RealtimeClient {
  constructor(handlers = {}) {
    this.h = handlers;
    this.ws = null;
    this.pingTimer = null;
    this.rttMs = null;
    this.droppedMs = 0;      // 上行因拥塞本地丢弃的音频
    this.upCapMs = 800;      // 上行待发上限：超过就不再塞入（丢最新）
  }

  // 浏览器发送缓冲区里还没发出去的音频时长（ms）
  bufferedMs() {
    if (!this.ws) return 0;
    return (this.ws.bufferedAmount / 4 / 16000) * 1000;
  }

  sendAudio(f32) {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    if (this.bufferedMs() > this.upCapMs) {
      // 发送侧已积压：丢弃这一小段（丢最新），避免把延迟越堆越高
      this.droppedMs += (f32.length / 16000) * 1000;
      return false;
    }
    this.ws.send(f32);
    return true;
  }

  /**
   * 注册音频上传：一次性整段发送，不套用实时流的发送背压上限。
   * 实时流用 upCapMs 丢帧是为了压延迟；注册是一次性 ≤8s 的数据，若也按 800ms 上限丢，
   * 整段会被截断成约 0.8s，后端只能判「时长不足」。
   */
  sendEnrollAudio(f32) {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(f32);
    return true;
  }

  connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl());
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      ws.onopen = () => {
        this._startPing();
        resolve();
      };
      ws.onerror = (e) => {
        this.h.onError?.("WebSocket 连接失败：请确认后端已启动");
        reject(e);
      };
      ws.onclose = () => {
        clearInterval(this.pingTimer);
        this.h.onClose?.();
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          let msg;
          try {
            msg = JSON.parse(ev.data);
          } catch {
            return;
          }
          if (msg.type === "pong") {
            this.rttMs = performance.now() - msg.t;
            this.h.onRtt?.(this.rttMs);
            return;
          }
          this.h.onJson?.(msg);
        } else {
          this.h.onAudio?.(new Float32Array(ev.data));
        }
      };
    });
  }

  _startPing() {
    const ping = () => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: "ping", t: performance.now() }));
      }
    };
    ping();
    this.pingTimer = setInterval(ping, 1500);
  }

  sendJson(obj) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj));
  }

  enroll(source = "mic") {
    this.sendJson({ type: "enroll", source });
  }

  start() {
    this.sendJson({ type: "start" });
  }

  stop() {
    this.sendJson({ type: "stop" });
  }

  /**
   * 运行时开关输出端后置降噪。服务端回 {"type":"config","post_denoise":...} 后才更新 UI，
   * 不做乐观更新（见 App.jsx）。
   */
  setPostDenoise(on) {
    this.sendJson({ type: "config", post_denoise: !!on });
  }

  close() {
    clearInterval(this.pingTimer);
    try {
      this.ws?.close();
    } catch {}
    this.ws = null;
  }
}
