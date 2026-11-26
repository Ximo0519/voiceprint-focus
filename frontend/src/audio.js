const TARGET_SR = 16000;
const DIAG_MAX_S = 60;
const BLOCK_S = 0.032;          // 服务端输出块长
const BUF_START_BLOCKS = 1.5;   // 起步 48ms
const BUF_MIN_BLOCKS = 1;       // 下限 32ms
const BUF_MAX_BLOCKS = 3;       // 封顶 96ms
const BUF_RISE_STEP = 0.5;      // 欠载 +0.5 块（48→64→80→96）
const BUF_FALL_STEP = 0.5;
const BUF_FALL_HOLD_S = 4.0;    // 连续 4s 无欠载才允许降档（3–5s 取中）
const BUF_FALL_P95_RATIO = 0.8; // 到达间隔 p95 < 0.8×目标档时长才降
const BUF_FADE_S = 0.005;       // 升档插空落在有声块时的 5ms 淡入
const SILENT_DB = -80;          // 块内 RMS 低于此值视为静音
const SILENCE_BLOCKS = 16;      // 0.5s

/**
 * 诊断导出（默认不启用）：只拷贝送进播放的 16k 块，不插入任何节点、不改动播放路径，
 * 因此不会影响输出音质。停止时由 AudioEngine.endDiag() 统一导出。
 */
class DiagCapture {
  constructor() {
    this.chunks = [];
    this.samples = 0;
    this.armed = false;
    this.truncated = false;
    this.tag = "";
  }

  start() {
    this.chunks = [];
    this.samples = 0;
    this.truncated = false;
    this.armed = true;
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    this.tag = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-`
      + `${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  }

  push(f32) {
    if (!this.armed) return;
    const cap = DIAG_MAX_S * TARGET_SR;
    if (this.samples >= cap) {
      this.truncated = true;
      return;
    }
    const c = f32.slice(0, Math.min(f32.length, cap - this.samples));
    this.chunks.push(c);
    this.samples += c.length;
  }

  stop() {
    this.armed = false;
    return { chunks: this.chunks, samples: this.samples, tag: this.tag, truncated: this.truncated };
  }
}

/** 块内 RMS（dBFS），用于判断是否为静音块。 */
function blockRmsDb(f32) {
  let s = 0;
  for (let i = 0; i < f32.length; i++) s += f32[i] * f32[i];
  return 10 * Math.log10(s / Math.max(1, f32.length) + 1e-20);
}

/**
 * 自适应播放缓冲：默认 1.5 块（48ms），欠载升档、慢降、封顶 3 块。
 *  - 快升：任意欠载 +0.5 块；升档插进去的空档若落在有声块，用 5ms 淡入掩掉；
 *  - 慢降：连续 4s 无欠载且最近到达间隔 p95 < 0.8×目标档，才降 0.5 块；
 *    降档不提前放出块，只在 ≥0.5s 静音且当前块已到达时把 nextTime 重锚到 now + 目标档；
 *  - 封顶：缓冲超前 > 3 块时跳到最新，丢最旧。
 */
export class BufferPolicy {
  constructor() { this.reset(); }

  reset() {
    this.target = BUF_START_BLOCKS;
    this.nextTime = 0;
    this.arrivals = [];
    this.silRun = 0;
    this.lastUnderT = -1e9;
    this.stats = {
      underruns: 0, underrunsOnVoiced: 0,
      rises: 0, falls: 0, riseOnVoiced: 0, fallOnVoiced: 0,
      caps: 0, reanchors: 0, reanchorOnVoiced: 0,
    };
  }

  _p95() {
    if (this.arrivals.length < 8) return 0;
    const d = [];
    for (let i = 1; i < this.arrivals.length; i++) {
      const v = this.arrivals[i] - this.arrivals[i - 1];
      if (v > 0) d.push(v);
    }
    if (d.length < 4) return 0;
    d.sort((a, b) => a - b);
    return d[Math.min(d.length - 1, Math.floor(0.95 * d.length))];
  }

  _tryFall(now, silent) {
    if (now - this.lastUnderT < BUF_FALL_HOLD_S || this.target <= BUF_MIN_BLOCKS) return;
    const p95 = this._p95();
    if (!(p95 > 0 && p95 < BUF_FALL_P95_RATIO * this.target * BLOCK_S)) return;
    this.target = Math.max(BUF_MIN_BLOCKS, +(this.target - BUF_FALL_STEP).toFixed(2));
    this.stats.falls += 1;
    if (!silent) this.stats.fallOnVoiced += 1;
    // 只改目标档；静音且下一块已到达时才重锚（绝不提前放出有声块）
    const t = now + this.target * BLOCK_S;
    if (this.silRun >= SILENCE_BLOCKS && this.nextTime > t) {
      this.nextTime = t;
      this.stats.reanchors += 1;
      if (!silent) this.stats.reanchorOnVoiced += 1;
    }
  }

  /** 新块到达：返回播放起点、插入的空档时长、是否需要淡入。 */
  arrive(now, dur, silent) {
    this.silRun = silent ? this.silRun + 1 : 0;
    this.arrivals.push(now);
    if (this.arrivals.length > 240) this.arrivals.shift();
    this._tryFall(now, silent);

    let gapMs = 0;
    let fadeIn = false;
    if (this.nextTime === 0) {
      this.nextTime = now + this.target * BLOCK_S;
    } else if (this.nextTime < now + 0.005) {
      gapMs = (now + this.target * BLOCK_S - this.nextTime) * 1000;
      this.stats.underruns += 1;
      if (!silent) this.stats.underrunsOnVoiced += 1;
      if (this.target < BUF_MAX_BLOCKS) {
        this.target = Math.min(BUF_MAX_BLOCKS, +(this.target + BUF_RISE_STEP).toFixed(2));
        this.stats.rises += 1;
        if (!silent) this.stats.riseOnVoiced += 1;
      }
      fadeIn = !silent;   // 插进去的空档落在有声块上：5ms 淡入掩掉
      this.nextTime = now + this.target * BLOCK_S;
      this.lastUnderT = now;
    }

    if (this.nextTime - now > BUF_MAX_BLOCKS * BLOCK_S) {
      this.nextTime = now + BUF_MAX_BLOCKS * BLOCK_S;
      this.stats.caps += 1;
    }

    const startAt = this.nextTime;
    this.nextTime += dur;
    return { startAt, gapMs, fadeIn };
  }
}


/** 线性插值降采样（仅在浏览器拒绝 16k AudioContext 时兜底）。 */
class LinearResampler {
  constructor(inRate, outRate) {
    this.ratio = inRate / outRate;
    this.pos = 0;
    this.buf = new Float32Array(0);
  }
  push(x) {
    const merged = new Float32Array(this.buf.length + x.length);
    merged.set(this.buf, 0);
    merged.set(x, this.buf.length);
    const out = [];
    let p = this.pos;
    while (p + 1 < merged.length) {
      const i = Math.floor(p);
      const f = p - i;
      out.push(merged[i] * (1 - f) + merged[i + 1] * f);
      p += this.ratio;
    }
    const consumed = Math.floor(p);
    this.buf = merged.slice(consumed);
    this.pos = p - consumed;
    return new Float32Array(out);
  }
}

export class AudioEngine {
  constructor() {
    this.ctx = null;
    this.outGain = null;
    this.nextTime = 0;
    this.policy = new BufferPolicy();
    this.sources = [];
    this.micStream = null;
    this.micNode = null;
    this.micSrc = null;
    this.resampler = null;
    this.underruns = 0;
    this.queuedSamples = 0;
    this.playedSamples = 0;
    this.diag = new DiagCapture();
  }

  async init() {
    if (this.ctx) return this.ctx;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx({ sampleRate: TARGET_SR, latencyHint: "interactive" });
    if (this.ctx.state === "suspended") await this.ctx.resume();
    this.outGain = this.ctx.createGain();
    this.outGain.gain.value = 1.0;
    this.outGain.connect(this.ctx.destination);
    return this.ctx;
  }

  get nativeRate() {
    return this.ctx ? this.ctx.sampleRate : TARGET_SR;
  }

  /** 麦克风 8ms 分块回调；若硬件采样率非 16k 则自动重采样。 */
  async startMic(onChunk) {
    await this.init();
    if (!this.workletReady) {
      await this.ctx.audioWorklet.addModule("/capture-worklet.js");
      this.workletReady = true;
    }
    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    this.micSrc = this.ctx.createMediaStreamSource(this.micStream);
    this.micNode = new AudioWorkletNode(this.ctx, "capture-processor");
    if (this.nativeRate !== TARGET_SR) {
      this.resampler = new LinearResampler(this.nativeRate, TARGET_SR);
    } else {
      this.resampler = null;
    }
    this.micNode.port.onmessage = (ev) => {
      const raw = new Float32Array(ev.data);
      const out = this.resampler ? this.resampler.push(raw) : raw;
      if (out.length) onChunk(out);
    };
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    this.micSrc.connect(this.micNode);
    this.micNode.connect(mute);
    mute.connect(this.ctx.destination);
    this.muteNode = mute;
  }

  stopMic() {
    try {
      this.micNode?.port?.close();
      this.micNode?.disconnect();
      this.muteNode?.disconnect();
      this.micSrc?.disconnect();
      this.micStream?.getTracks().forEach((t) => t.stop());
    } catch {}
    this.micStream = null;
    this.micNode = null;
    this.micSrc = null;
    this.muteNode = null;
  }

  /** 排入服务端返回的 16k 输出块；自适应缓冲（默认 1.5 块 = 48ms，封顶 3 块）。 */
  enqueueOut(f32) {
    if (!this.ctx || !f32.length) return;
    this.diag.push(f32);
    const now = this.ctx.currentTime;
    const r = this.policy.arrive(now, f32.length / TARGET_SR, blockRmsDb(f32) < SILENT_DB);
    const buf = this.ctx.createBuffer(1, f32.length, TARGET_SR);
    buf.copyToChannel(f32, 0);
    if (r.fadeIn) {
      const ch = buf.getChannelData(0);
      const n = Math.min(ch.length, Math.round(BUF_FADE_S * TARGET_SR));
      for (let i = 0; i < n; i++) ch[i] *= i / n;
    }
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.outGain);
    src.start(r.startAt);
    this.nextTime = this.policy.nextTime;
    this.underruns = this.policy.stats.underruns;
    this.queuedSamples += f32.length;
    this.playedSamples += f32.length;
    this.sources.push(src);
    if (this.sources.length > 256) this.sources.splice(0, this.sources.length - 256);
  }

  /** 一次性播放（A/B 试听）。 */
  async playBuffer(f32) {
    await this.init();
    if (!f32 || !f32.length) return;
    this.stopPlayback();
    const buf = this.ctx.createBuffer(1, f32.length, TARGET_SR);
    buf.copyToChannel(f32, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.outGain);
    src.start();
    this.abSource = src;
  }

  stopPlayback() {
    try {
      this.abSource?.stop();
    } catch {}
    this.abSource = null;
  }

  resetQueue() {
    this.policy.reset();
    this.nextTime = this.ctx ? this.ctx.currentTime + 0.02 : 0;
    this.queuedSamples = 0;
    this.underruns = 0;
  }

  queueDelayMs() {
    if (!this.ctx) return 0;
    return Math.max(0, (this.policy.nextTime - this.ctx.currentTime) * 1000);
  }

  /** 缓冲策略统计（档位、欠载、升降/重锚/封顶次数）。 */
  bufferStats() {
    return {
      targetBlocks: this.policy.target,
      bufferMs: this.queueDelayMs(),
      underruns: this.policy.stats.underruns,
      ...this.policy.stats,
    };
  }

  /** 诊断导出：开始/停止与串流开始/停止绑定；未启用时这两个方法不会被调用。 */
  beginDiag() {
    this.diag.start();
  }

  async endDiag() {
    const d = this.diag.stop();
    if (!d.samples) return null;
    const recv = concatFloat32(d.chunks);
    const play = await this.render48k(d.chunks);
    return {
      recv,
      play,
      meta: {
        schema: "vf-diag/1",
        tag: d.tag,
        created_at: new Date().toISOString(),
        recv16k: { sr: TARGET_SR, samples: recv.length, dur_s: +(recv.length / TARGET_SR).toFixed(3) },
        play48k: { sr: 48000, samples: play.length, dur_s: +(play.length / 48000).toFixed(3) },
        live_ctx_rate: this.nativeRate,
        truncated: d.truncated,
        max_s: DIAG_MAX_S,
        chunks: d.chunks.length,
        method: "recv16k=服务端下发原样 PCM；play48k=把同一批 16k 样本拼成连续流后用 OfflineAudioContext@48k 渲染",
        note: "实时 AudioContext 为 16k，16k→48k 的设备端重采样在 JS 之外、且作用于连续流；此处用 Web Audio 重采样器对完全相同的连续音频渲染 48k，块边界不参与重采样。",
      },
    };
  }

  /**
   * 用 Web Audio 的 16k→48k 重采样器渲染同一段音频。
   * 实时播放的 AudioContext 是 16k，设备端重采样在 JS 之外且作用于连续流，
   * 因此这里把块拼成连续信号、只用一个 source 渲染，避免逐块重采样引入边界伪影。
   */
  async render48k(chunks) {
    const rate = 48000;
    const total = chunks.reduce((a, c) => a + c.length, 0);
    const n = Math.max(1, Math.ceil((total * rate) / TARGET_SR));
    const off = new OfflineAudioContext(1, n, rate);
    const b = off.createBuffer(1, total, TARGET_SR);
    b.copyToChannel(concatFloat32(chunks), 0);
    const s = off.createBufferSource();
    s.buffer = b;
    s.connect(off.destination);
    s.start(0);
    const rendered = await off.startRendering();
    return new Float32Array(rendered.getChannelData(0));
  }

  /** 解码任意采样率的音频并重采样到 16k。 */
  async decodeTo16k(arrayBuffer) {
    await this.init();
    const tmp = await this.ctx.decodeAudioData(arrayBuffer.slice(0));
    const src = tmp.getChannelData(0);
    if (Math.abs(tmp.sampleRate - TARGET_SR) < 1) return new Float32Array(src);
    const n = Math.round((src.length * TARGET_SR) / tmp.sampleRate);
    const off = new OfflineAudioContext(1, n, TARGET_SR);
    const b = off.createBuffer(1, src.length, tmp.sampleRate);
    b.copyToChannel(src, 0);
    const s = off.createBufferSource();
    s.buffer = b;
    s.connect(off.destination);
    s.start();
    const rendered = await off.startRendering();
    return new Float32Array(rendered.getChannelData(0));
  }
}

export function encodeWav(f32, sr = TARGET_SR) {
  const n = f32.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const str = (off, s) => [...s].forEach((c, i) => dv.setUint8(off + i, c.charCodeAt(0)));
  str(0, "RIFF");
  dv.setUint32(4, 36 + n * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true);
  dv.setUint32(28, sr * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  str(36, "data");
  dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, f32[i]));
    dv.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

export function concatFloat32(chunks) {
  const total = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}
