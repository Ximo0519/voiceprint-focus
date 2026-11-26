import { useCallback, useEffect, useRef, useState } from "react";
import { RealtimeClient } from "./api";
import { AudioEngine, concatFloat32, encodeWav } from "./audio";

const SR = 16000;

// 长流保护：指标/曲线只保留滑窗，导出音频设上限。
// 修复前这些数组全程累加，且每 300ms 对全量做 sort + Math.max(...arr) 展开，
// 约 35–70 分钟就会因 JS 实参上限抛 RangeError，指标面板从此不再刷新。
const MAX_STATS = 1800;            // ≈ 1 分钟（32ms 一块）
const MAX_QUEUE_SAMPLES = 600;     // ≈ 3 分钟（300ms 一点）
const MAX_TRACK_SAMPLES = 600 * SR; // 导出音频上限 10 分钟（16k 单声道约 38MB）
const maxOf = (arr) => arr.reduce((m, v) => (v > m ? v : m), -Infinity);
// 推入采集块并维护样本计数，超过上限后不再增长（避免长会议把内存打满）
const pushChunk = (ref, counterRef, chunk) => {
  if (counterRef.current >= MAX_TRACK_SAMPLES) return;
  ref.current.push(chunk);
  counterRef.current += chunk.length;
};

function makeWave(n = 4096) {
  return { data: new Float32Array(n), pos: 0 };
}
function pushWave(store, chunk) {
  const { data } = store;
  let p = store.pos;
  for (let i = 0; i < chunk.length; i++) {
    data[p] = chunk[i];
    p = (p + 1) % data.length;
  }
  store.pos = p;
}
function clearWave(store) {
  store.data.fill(0);
  store.pos = 0;
}
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

function Waveform({ store, color, label, height = 56 }) {
  const ref = useRef(null);
  useEffect(() => {
    let raf;
    const draw = () => {
      const cv = ref.current;
      if (cv) {
        const dpr = window.devicePixelRatio || 1;
        const w = cv.clientWidth;
        const h = height;
        if (cv.width !== w * dpr || cv.height !== h * dpr) {
          cv.width = w * dpr;
          cv.height = h * dpr;
        }
        const ctx = cv.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.9;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        const { data, pos } = store;
        const n = data.length;
        for (let x = 0; x < w; x++) {
          const i = Math.floor((x / w) * n);
          const v = data[(pos + i) % n];
          const y = h / 2 - v * h * 0.46;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 0.18;
        ctx.fillStyle = color;
        ctx.fillRect(0, h / 2 - 0.5, w, 1);
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [store, color, height]);
  return (
    <div className="wave-wrap">
      <span className="wave-label">{label}</span>
      <canvas ref={ref} style={{ width: "100%", height }} />
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState("demo");
  const [phase, setPhase] = useState("idle");
  const [demos, setDemos] = useState([]);
  const [demoId, setDemoId] = useState("");
  const [health, setHealth] = useState(null);
  const [notice, setNotice] = useState(null);
  const [vad, setVad] = useState(null);
  const [ready, setReady] = useState(null);
  const [enrollSec, setEnrollSec] = useState(0);
  const [progress, setProgress] = useState(0);
  const [summary, setSummary] = useState(null);
  const [metrics, setMetrics] = useState({});
  const [rtt, setRtt] = useState(null);
  const [abUrls, setAbUrls] = useState({ mix: null, out: null });
  const [diagExport, setDiagExport] = useState(false);
  // 后置降噪：开流前 / 开流中都可切换，作用于所有活跃会话；
  // 以服务端 /api/config 回读值为准，不做乐观更新
  const [pdOn, setPdOn] = useState(null);
  const [pdFloor, setPdFloor] = useState(null);
  const [pdRampMs, setPdRampMs] = useState(128);
  const [pdSessions, setPdSessions] = useState(null);
  const [pdBusy, setPdBusy] = useState(false);

  const engineRef = useRef(null);
  const clientRef = useRef(null);
  const destRef = useRef("hold"); // enroll | stream | hold
  const enrollBufRef = useRef([]);
  const inputChunksRef = useRef([]);
  const outputChunksRef = useRef([]);
  const statsRef = useRef([]);
  const queueSamplesRef = useRef([]);
  const chunkCountRef = useRef(0);
  const audioSRef = useRef(0);
  const computeMsRef = useRef(0);
  const overrunsRef = useRef(0);
  const inSamplesRef = useRef(0);
  const outSamplesRef = useRef(0);
  const abUrlsRef = useRef({ mix: null, out: null });
  const waveIn = useRef(makeWave());
  const waveOut = useRef(makeWave());
  const feedRef = useRef(null);
  const demoMixRef = useRef(null);
  const enrollStopRef = useRef(null);
  const fileRef = useRef(null);
  const phaseRef = useRef(phase);
  const modeRef = useRef(mode);
  const readyRef = useRef(false);

  phaseRef.current = phase;
  modeRef.current = mode;

  const engine = () => {
    if (!engineRef.current) engineRef.current = new AudioEngine();
    return engineRef.current;
  };

  async function refreshConfig() {
    try {
      const r = await fetch("/api/config");
      if (!r.ok) return;
      const c = await r.json();
      if (typeof c.post_denoise === "boolean") setPdOn(c.post_denoise);
      if (c.floor_db != null) setPdFloor(c.floor_db);
      if (c.ramp_blocks != null) setPdRampMs(c.ramp_blocks * 32);
      if (c.active_sessions != null) setPdSessions(c.active_sessions);
    } catch {
      // 回读失败不打断会话；切换失败会由开关操作本身提示
    }
  }

  // 阶段变化（会话建立 / 结束）时回读运行期配置
  useEffect(() => {
    void refreshConfig();
  }, [phase]);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => {
        setHealth(h);
        if (typeof h.post_denoise === "boolean") setPdOn(h.post_denoise);
        if (h.post_denoise_floor_db != null) setPdFloor(h.post_denoise_floor_db);
        if (h.post_denoise_ramp_blocks != null) setPdRampMs(h.post_denoise_ramp_blocks * 32);
      })
      .catch(() => setHealth({ ok: false }));
    fetch("/api/demo/list")
      .then((r) => r.json())
      .then((list) => {
        setDemos(list);
        if (list.length) setDemoId(list[0].id);
      })
      .catch(() => {});
  }, []);

  // 流式阶段每 300ms 刷新指标（避免每条 stats 都触发 React 重渲染）
  useEffect(() => {
    if (phase !== "streaming" && phase !== "stopping") return;
    const t = setInterval(() => {
      const arr = statsRef.current;
      const comp = arr.map((s) => s.compute_ms);
      const algo = arr.map((s) => s.algo_latency_ms);
      const eng = engineRef.current;
      const q = eng ? eng.queueDelayMs() : 0;
      queueSamplesRef.current.push(q);
      if (queueSamplesRef.current.length > MAX_QUEUE_SAMPLES) {
        queueSamplesRef.current.splice(
          0, queueSamplesRef.current.length - MAX_QUEUE_SAMPLES);
      }
      const qs = [...queueSamplesRef.current].sort((a, b) => a - b);
      const algoSorted = [...algo].sort((a, b) => a - b);
      const last = arr.length ? arr[arr.length - 1] : null;
      const srvQueue = last?.queue_in_ms ?? null;
      setMetrics({
        chunks: chunkCountRef.current,
        computeMean: comp.length ? comp.reduce((a, b) => a + b, 0) / comp.length : null,
        computeP95: comp.length
          ? [...comp].sort((a, b) => a - b)[Math.floor(comp.length * 0.95)]
          : null,
        computeMax: comp.length ? maxOf(comp) : null,
        algoLast: algo.length ? algo[algo.length - 1] : null,
        algoMed: algo.length ? algoSorted[Math.floor(algoSorted.length / 2)] : null,
        algoMax: algo.length ? maxOf(algo) : null,
        rtf: audioSRef.current > 0 ? computeMsRef.current / 1000 / audioSRef.current : null,
        queue: qs.length ? qs[Math.floor(qs.length / 2)] : 0,
        queueNow: q,
        underruns: eng ? eng.underruns : 0,
        audioS: audioSRef.current,
        srvQueue,
        srvQueueMax: arr.length ? maxOf(arr.map((s) => s.queue_in_ms ?? 0)) : null,
        srvOut: last?.queue_out_ms ?? null,
        upBuf: clientRef.current?.bufferedMs?.() ?? 0,
        dropIn: last?.dropped_in_ms ?? 0,
        dropOut: last?.dropped_out_ms ?? 0,
        skipped: last?.skipped_ms ?? 0,
        overruns: overrunsRef.current,
        clientDrop: clientRef.current?.droppedMs ?? 0,
      });
    }, 300);
    return () => clearInterval(t);
  }, [phase]);

  const ensureClient = useCallback(async () => {
    if (clientRef.current) return clientRef.current;
    const c = new RealtimeClient({
      onJson: (msg) => {
        if (msg.type === "ready") {
          readyRef.current = true;
          setReady(msg);
          setVad(msg.vad);
          setPhase("streaming");
          setNotice(null);
          if (msg.model && typeof msg.model.post_denoise === "boolean") setPdOn(msg.model.post_denoise);
          if (msg.model && msg.model.post_denoise_floor_db != null) setPdFloor(msg.model.post_denoise_floor_db);
          setPdBusy(false);
          statsRef.current = [];
          queueSamplesRef.current = [];
          inputChunksRef.current = [];
          outputChunksRef.current = [];
          chunkCountRef.current = 0;
          audioSRef.current = 0;
          computeMsRef.current = 0;
          overrunsRef.current = 0;
          inSamplesRef.current = 0;
          outSamplesRef.current = 0;
          clearWave(waveIn.current);
          clearWave(waveOut.current);
          engine().resetQueue();
          if (modeRef.current === "demo") startDemoFeed();
          else destRef.current = "stream";
        } else if (msg.type === "config") {
          // 兼容 WS 侧回读（界面切换走 REST /api/config）
          if (typeof msg.post_denoise === "boolean") setPdOn(msg.post_denoise);
          setPdBusy(false);
        } else if (msg.type === "stats") {
          chunkCountRef.current += 1;
          audioSRef.current += (msg.frames || 0) * 0.008;
          computeMsRef.current += msg.compute_ms || 0;
          if (msg.overrun) overrunsRef.current += 1;
          const arr = statsRef.current;
          arr.push(msg);
          if (arr.length > MAX_STATS) arr.splice(0, arr.length - MAX_STATS);
        } else if (msg.type === "done") {
          readyRef.current = false;
          setSummary(msg.summary);
          setPhase("done");
          finishSession();
        } else if (msg.type === "error") {
          readyRef.current = false;
          setPdBusy(false);
          setPhase("error");
          setNotice({ kind: "error", text: msg.message, vad: msg.vad, qc: msg.enroll_qc });
          finishSession();
        }
      },
      onAudio: (f32) => {
        engine().enqueueOut(f32);
        pushChunk(outputChunksRef, outSamplesRef, f32);
        pushWave(waveOut.current, f32);
      },
      onError: () => setNotice({ kind: "error", text: "无法连接后端服务，请确认已启动" }),
      onClose: () => {
        readyRef.current = false;
        clientRef.current = null;
        setPdBusy(false);
        if (phaseRef.current === "streaming") {
          setPhase("error");
          setNotice({ kind: "error", text: "连接已断开" });
        }
      },
      onRtt: setRtt,
    });
    await c.connect();
    clientRef.current = c;
    return c;
  }, []);

  function finishSession() {
    destRef.current = "hold";
    if (feedRef.current) {
      clearTimeout(feedRef.current);
      feedRef.current = null;
    }
    enrollStopRef.current?.();
    engine().stopMic();
    void finishDiagExport();
    const eng = engineRef.current;
    if (eng) setMetrics((m) => ({ ...m, underruns: eng.underruns, queueNow: eng.queueDelayMs() }));
  }

  async function togglePostDenoise() {
    if (pdBusy || pdOn == null) return;
    const next = !pdOn;
    setPdBusy(true);
    try {
      const r = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ post_denoise: next }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      // 只认服务端回读值：开流前与开流中共用同一条全局配置
      if (typeof data.post_denoise === "boolean") setPdOn(data.post_denoise);
      if (data.floor_db != null) setPdFloor(data.floor_db);
      if (data.ramp_ms != null) setPdRampMs(data.ramp_ms);
      if (data.active_sessions != null) setPdSessions(data.active_sessions);
    } catch (e) {
      setNotice({ kind: "error", text: `后置降噪切换失败：${e.message || e}` });
    } finally {
      setPdBusy(false);
    }
  }

  async function finishDiagExport() {
    const eng = engineRef.current;
    if (!eng || !eng.diag.armed) return;
    try {
      const r = await eng.endDiag();
      if (!r) return;
      const dl = (blob, name) => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      };
      dl(encodeWav(r.recv, 16000), `diag_${r.meta.tag}_recv16k.wav`);
      dl(encodeWav(r.play, 48000), `diag_${r.meta.tag}_play48k.wav`);
      dl(new Blob([JSON.stringify(r.meta, null, 2)], { type: "application/json" }),
        `diag_${r.meta.tag}_meta.json`);
      setNotice({ kind: "ok", text: "诊断导出已开始下载（若提示多文件下载请允许）" });
    } catch (e) {
      setNotice({ kind: "error", text: `诊断导出失败：${e.message || e}` });
    }
  }

  async function connectOnly() {
    setNotice(null);
    try {
      await ensureClient();
      setNotice({ kind: "ok", text: "服务已连接" });
    } catch {
      setNotice({ kind: "error", text: "连接失败：请先启动后端（uvicorn server:app）" });
    }
  }

  // ---------------- 麦克风模式 ----------------
  async function startEnrollMic() {
    setNotice(null);
    setSummary(null);
    setReady(null);
    setVad(null);
    setEnrollSec(0);
    enrollBufRef.current = [];
    try {
      await ensureClient();
      await engine().init();
      engine().resetQueue();
      if (diagExport) engine().beginDiag();
      destRef.current = "enroll";
      setPhase("enroll-recording");
      let stopped = false;
      enrollStopRef.current = () => {
        stopped = true;
      };
      await engine().startMic((chunk) => {
        if (destRef.current === "enroll") {
          enrollBufRef.current.push(chunk);
          pushWave(waveIn.current, chunk);
          const s = enrollBufRef.current.reduce((a, c) => a + c.length, 0) / SR;
          setEnrollSec(s);
          if (s >= 5.0 && !stopped) {
            stopped = true;
            finishEnroll();
          }
        } else if (destRef.current === "stream") {
          clientRef.current?.sendAudio(chunk);
        pushChunk(inputChunksRef, inSamplesRef, chunk);
          pushWave(waveIn.current, chunk);
        }
      });
    } catch (e) {
      setPhase("error");
      setNotice({ kind: "error", text: `麦克风不可用：${e.message || e}` });
    }
  }

  async function finishEnroll() {
    if (destRef.current !== "enroll") return;
    destRef.current = "hold";
    const buf = concatFloat32(enrollBufRef.current);
    const secs = buf.length / SR;
    if (secs < 2.5 || secs > 6.5) {
      setPhase("error");
      setNotice({
        kind: "error",
        text: `注册音频 ${secs.toFixed(1)}s，需在 2.5–6.5s 之间，请重录`,
      });
      return;
    }
    setPhase("enrolling");
    const c = await ensureClient();
    c.enroll();
    for (let i = 0; i < buf.length; i += 1600) c.sendEnrollAudio(buf.slice(i, i + 1600));
    c.start();
  }

  // ---------------- 上传音频注册（与录制同一条注册路径） ----------------
  async function startEnrollFile(ev) {
    const f = ev.target.files && ev.target.files[0];
    ev.target.value = "";                 // 允许重复选同一个文件
    if (!f) return;
    setNotice(null);
    setSummary(null);
    setReady(null);
    setVad(null);
    setEnrollSec(0);
    setPhase("loading");
    try {
      await ensureClient();
      const eng = engine();
      await eng.init();
      eng.resetQueue();
      if (diagExport) eng.beginDiag();
      // 前端解码任意格式，统一重采样成 16k 单声道，与服务端录制路径完全一致
      const buf = await eng.decodeTo16k(await f.arrayBuffer());
      const secs = buf.length / SR;
      if (secs < 2.5 || secs > 6.5) {
        setPhase("error");
        setNotice({ kind: "error", text: `文件时长 ${secs.toFixed(1)}s，需在 2.5–6.5s 之间` });
        return;
      }
      clearWave(waveIn.current);
      clearWave(waveOut.current);
      const c = clientRef.current;
      c.enroll("upload");                 // 只多带一个来源标记，判定逻辑不变
      for (let i = 0; i < buf.length; i += 1600) c.sendEnrollAudio(buf.slice(i, i + 1600));
      c.start();
      setPhase("enrolling");
      setNotice({ kind: "ok", text: `已载入 ${f.name}（${secs.toFixed(1)}s，转 16k 单声道）` });
      // 注册期间 dest="hold" 丢弃麦克风音频；ready 后自动切到 stream 做实时试听
      destRef.current = "hold";
      try {
        engine().stopMic();          // 避免上一轮麦克风仍在采音形成双路
        await engine().startMic((chunk) => {
          if (destRef.current === "stream") {
            clientRef.current?.sendAudio(chunk);
          pushChunk(inputChunksRef, inSamplesRef, chunk);
            pushWave(waveIn.current, chunk);
          }
        });
      } catch (e) {
        setNotice({ kind: "error", text: `声纹已提交，但麦克风不可用，无法试听：${e.message || e}` });
      }
    } catch (e) {
      setPhase("error");
      setNotice({ kind: "error", text: `上传注册失败：${e.message || e}` });
    }
  }

  // ---------------- 演示素材模式 ----------------
  async function startDemo() {
    const item = demos.find((d) => d.id === demoId);
    if (!item) return;
    setNotice(null);
    setSummary(null);
    setReady(null);
    setVad(null);
    setProgress(0);
    setPhase("loading");
    try {
      await ensureClient();
      const eng = engine();
      await eng.init();
      if (diagExport) eng.beginDiag();
      const [mixBuf, enrBuf] = await Promise.all([
        fetch(item.mix).then((r) => r.arrayBuffer()),
        fetch(item.enroll).then((r) => r.arrayBuffer()),
      ]);
      const mix = await eng.decodeTo16k(mixBuf);
      const enr = await eng.decodeTo16k(enrBuf);
      demoMixRef.current = mix;
      clearWave(waveIn.current);
      clearWave(waveOut.current);
      const c = clientRef.current;
      c.enroll();
      for (let i = 0; i < enr.length; i += 1600) c.sendEnrollAudio(enr.slice(i, i + 1600));
      c.start();
    } catch (e) {
      setPhase("error");
      setNotice({ kind: "error", text: `载入演示素材失败：${e.message || e}` });
    }
  }

  function startDemoFeed() {
    const mix = demoMixRef.current;
    if (!mix) return;
    destRef.current = "stream";
    engine().resetQueue();
    const t0 = performance.now();
    let sent = 0;
    const tick = () => {
      if (destRef.current !== "stream") return;
      const elapsed = performance.now() - t0;
      const target = Math.min(mix.length, Math.floor((elapsed / 1000) * SR));
      if (target > sent) {
        let pos = sent;
        while (pos < target) {
          const n = Math.min(128, target - pos);
          const chunk = mix.slice(pos, pos + n);
          clientRef.current?.sendAudio(chunk);
        pushChunk(inputChunksRef, inSamplesRef, chunk);
          pushWave(waveIn.current, chunk);
          pos += n;
        }
        sent = target;
        setProgress(sent / mix.length);
      }
      if (sent >= mix.length) {
        stopStreaming();
        return;
      }
      feedRef.current = setTimeout(tick, 10);
    };
    feedRef.current = setTimeout(tick, 10);
  }

  function stopStreaming() {
    if (destRef.current === "hold") return;
    destRef.current = "hold";
    if (feedRef.current) {
      clearTimeout(feedRef.current);
      feedRef.current = null;
    }
    engine().stopMic();
    setPhase("stopping");
    clientRef.current?.stop();
  }

  // ---------------- 结果导出 ----------------
  useEffect(() => {
    if (phase !== "done") return;
    const inp = concatFloat32(inputChunksRef.current);
    const out = concatFloat32(outputChunksRef.current);
    if (!inp.length) return;
    const next = { mix: URL.createObjectURL(encodeWav(inp)), out: URL.createObjectURL(encodeWav(out)) };
    // 先释放上一轮的 URL，否则每完成一次会话就泄漏两份完整音频
    Object.values(abUrlsRef.current).forEach((u) => u && URL.revokeObjectURL(u));
    abUrlsRef.current = next;
    setAbUrls(next);
  }, [phase]);

  useEffect(() => () => {
    Object.values(abUrlsRef.current).forEach((u) => u && URL.revokeObjectURL(u));
  }, []);

  const inputSec = inputChunksRef.current.reduce((a, c) => a + c.length, 0) / SR;
  const outSec = outputChunksRef.current.reduce((a, c) => a + c.length, 0) / SR;

  const e2e =
    metrics.algoMed != null
      ? metrics.algoMed + (rtt ? rtt / 2 : 0) + (metrics.queue || 0) +
        (metrics.srvQueue || 0) + (metrics.upBuf || 0)
      : null;

  const busy = ["loading", "enroll-recording", "enrolling", "streaming", "stopping"].includes(phase);
  const connected = !!clientRef.current;

  return (
    <div className="app">
      <div className="grain" aria-hidden="true" />
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            VPR<span className="slash">//</span>FOCUS
          </div>
          <div className="brand-sub">声纹聚焦</div>
          <div className="brand-en">REALTIME VOICE EXTRACTION</div>
        </div>
        <div className="sysline">
          <div className="badges">
            <span className={`badge ${health?.ok ? "ok" : "bad"}`}>
              {health?.ok ? "服务在线" : "服务未连接"}
            </span>
            <span className="badge">ORT {health?.ort || "–"} · CPU</span>
            <span className="badge">16 kHz / win 512 / hop 128</span>
            <span className="badge">chunk 4 帧 = 32 ms</span>
            {rtt != null && <span className="badge">RTT {fmt(rtt, 1)} ms</span>}
          </div>
        </div>
      </header>

      <main className="stage">
        <header className="vhead">
          <p className="vhead-sec">实验16 · B2 帧级 TF-Map 条件化因果 BSRNN</p>
          <h1>
            声纹聚焦<em> · 32ms 实时流式</em>
          </h1>
          <p className="sub">
            先用 3–5 秒注册目标说话人，再对麦克风/演示音频做逐 32ms 流式选择性提取。
            监听 + 录制模式，不做全双工。
          </p>
        </header>

      {notice && (
        <div className={`notice ${notice.kind}`}>
          <span>{notice.text}</span>
          {notice.vad && (
            <span className="vad-inline">
              时长 {notice.vad.duration_s}s · 语音占比 {notice.vad.speech_ratio} · 峰值{" "}
              {notice.vad.peak} · 估计 SNR {notice.vad.snr_db}dB
            </span>
          )}
          {notice.qc && (
            <span className="vad-inline">
              同人一致性 cos {notice.qc.identity.cos}（阈值 {notice.qc.identity.cos_min}）· 音高差{" "}
              {notice.qc.pitch.checked
                ? `${notice.qc.pitch.shift_st} 半音（上限 ${notice.qc.pitch.shift_max_st}）`
                : "未测（基频不可靠）"}
            </span>
          )}
          <button onClick={() => setNotice(null)}>×</button>
        </div>
      )}

      <div className="grid">
        <section className="card">
          <div className="card-head">
            <h2>1 · 输入</h2>
            <div className="seg">
              <button className={mode === "demo" ? "on" : ""} onClick={() => !busy && setMode("demo")}>
                演示素材
              </button>
              <button className={mode === "mic" ? "on" : ""} onClick={() => !busy && setMode("mic")}>
                麦克风
              </button>
            </div>
          </div>

          {mode === "demo" ? (
            <>
              <label className="field">
                <span>内置演示（不同音色 + 0dB 单干扰，TTS 语料）</span>
                <select value={demoId} onChange={(e) => setDemoId(e.target.value)} disabled={busy}>
                  {demos.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.id} · {d.name} · SIR {d.sir_db}dB
                    </option>
                  ))}
                </select>
              </label>
              <div className="row">
                <button className="primary" onClick={startDemo} disabled={busy || !demos.length}>
                  载入素材并开始流式
                </button>
                <button onClick={connectOnly} disabled={busy}>
                  {connected ? "重新连接" : "连接服务"}
                </button>
              </div>
              <p className="hint">
                素材以 8ms 块按真实时间节奏推送到同一 WebSocket；目标声纹取自同一音色的另一条语句。
              </p>
            </>
          ) : (
            <>
              <div className="row">
                <button className="primary" onClick={startEnrollMic} disabled={busy}>
                  录制声纹（3–5 秒）
                </button>
                <button onClick={finishEnroll} disabled={phase !== "enroll-recording"}>
                  完成并开始
                </button>
                <button onClick={() => fileRef.current?.click()} disabled={busy}>
                  上传音频注册
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac"
                  hidden
                  onChange={startEnrollFile}
                />
                <button onClick={stopStreaming} disabled={phase !== "streaming"}>
                  停止
                </button>
              </div>
              <div className="enroll-bar">
                <div className="enroll-fill" style={{ width: `${Math.min(100, (enrollSec / 5) * 100)}%` }} />
                <span>{fmt(enrollSec, 1)}s / 5s</span>
              </div>
              <p className="hint">
                请用耳机监听，避免处理后的声音再次被麦克风采集。注册会依次校验：
                ① 前后半段是不是同一个人（多人混录会被拒）；② 前后半段音高是否一致（忽高忽低会被拒）；
                ③ 就绪后先说一句话做试听确认。保持一个人、平时音调、连贯说完。
                「上传音频注册」走同一条路径：文件在前端解码并重采样为 16k 单声道后按 8ms 块送入，判定与录制完全一致。
              </p>
            </>
          )}

          <Waveform store={waveIn.current} color="#0b96ad" label="输入" />
        </section>

        <section className="card">
          <div className="card-head">
            <h2>2 · 实时输出</h2>
            <span className={`state state-${phase}`}>{phaseLabel(phase)}</span>
          </div>
          <Waveform store={waveOut.current} color="#d62f6e" label="输出" />
          <div className="pd-row">
            <button
              type="button"
              role="switch"
              aria-checked={!!pdOn}
              aria-label="输出后置降噪开关"
              className={`switch ${pdOn ? "on" : ""}`}
              title="立即生效：作用于当前与后续会话"
              disabled={pdBusy || pdOn == null}
              onClick={togglePostDenoise}
            >
              <span className="slider" />
            </button>
            <div className="pd-text">
              <b>输出后置降噪 {pdOn == null ? "读取中…" : pdOn ? "开启" : "关闭"}</b>
              <span>
                门限 {pdFloor != null ? `${pdFloor} dB` : "−18 dB"} · 过渡{" "}
                {Math.round(pdRampMs / 32)} 块 / {pdRampMs} ms
                {pdSessions != null ? ` · 活跃会话 ${pdSessions}` : ""}
                {pdBusy ? " · 正在同步…" : ""}
              </span>
            </div>
          </div>
          <p className="hint">
            语音场景保持开启；音乐/播客类内容建议关闭（会削弱高频）。开流前与流式进行中都可切换，
            点击后立即作用于所有活跃会话，切换以 {Math.round(pdRampMs / 32)} 块（{pdRampMs}ms）淡入淡出过渡。
          </p>
          <div className="enroll-bar slim">
            <div className="enroll-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
            <span>
              进度 {Math.round(progress * 100)}% · 输入 {fmt(inputSec, 1)}s / 输出 {fmt(outSec, 1)}s
            </span>
          </div>
          <div className="row">
            <label className="check-inline">
              <input
                type="checkbox"
                checked={diagExport}
                onChange={(e) => setDiagExport(e.target.checked)}
                disabled={busy}
              />
              诊断导出（16k/48k）
            </label>
            <button onClick={stopStreaming} disabled={phase !== "streaming"}>
              停止并收尾
            </button>
          </div>
          {ready && (
            <div className="ready">
              <div>
                <b>声纹已就绪</b> · 注册 {ready.enroll_s}s · 编码 {ready.enroll_ms}ms
                {ready.enroll_source ? ` · 来源 ${ready.enroll_source === "upload" ? "上传文件" : "麦克风录制"}` : ""}
              </div>
              {ready.enroll_saved && (
                <div className="hint">注册音频已留档：{ready.enroll_saved}</div>
              )}
              <div className="kv">
                <span>VAD 时长</span>
                <b>{vad?.duration_s}s</b>
                <span>语音占比</span>
                <b>{vad?.speech_ratio}</b>
                <span>峰值</span>
                <b>{vad?.peak}</b>
                <span>估计 SNR</span>
                <b>{vad?.snr_db}dB</b>
              </div>
              {ready.enroll_qc && (
                <div className="kv">
                  <span>同人一致性</span>
                  <b>
                    cos {ready.enroll_qc.identity.cos}（阈值 {ready.enroll_qc.identity.cos_min}）
                  </b>
                  <span>音高差</span>
                  <b>
                    {ready.enroll_qc.pitch.checked
                      ? `${ready.enroll_qc.pitch.shift_st} 半音（上限 ${ready.enroll_qc.pitch.shift_max_st}）`
                      : "未测（基频不可靠）"}
                  </b>
                </div>
              )}
              <div className="verify">
                <b>试听确认 · 就绪后三步</b>
                <ol>
                  <li>一个人对着麦克风完整说一句 3 秒左右的话。</li>
                  <li>听“输出”里剩下的是不是你的声音：有别人残留、或明显不像，说明这次注册不可用。</li>
                  <li>不像就点“录制声纹”重录：换安静环境、保持平时音调、一个人连贯说完。</li>
                </ol>
              </div>
            </div>
          )}
          {phase === "done" && (
            <div className="ab">
              <span>A/B：</span>
              <button onClick={() => engine().playBuffer(concatFloat32(inputChunksRef.current))} disabled={!abUrls.mix}>
                原始混音
              </button>
              <button onClick={() => engine().playBuffer(concatFloat32(outputChunksRef.current))} disabled={!abUrls.out}>
                处理后
              </button>
              {abUrls.mix && (
                <a href={abUrls.mix} download="mix.wav">
                  下载原始 WAV
                </a>
              )}
              {abUrls.out && (
                <a href={abUrls.out} download="separated.wav">
                  下载处理 WAV
                </a>
              )}
            </div>
          )}
        </section>

        <section className="card metrics">
          <div className="card-head">
            <h2>3 · 实测指标</h2>
            <span className="hint-inline">目标端到端 ≤ 100ms</span>
          </div>
          <div className="metric-grid">
            <Metric label="实时端到端估计" value={e2e} unit="ms" good={e2e != null && e2e <= 100} />
            <Metric label="算法延迟（服务端稳态）" value={metrics.algoMed} unit="ms" />
            <Metric label="服务端输入积压" value={metrics.srvQueue} unit="ms" good={metrics.srvQueue != null && metrics.srvQueue <= 200} />
            <Metric label="服务端发送积压" value={metrics.srvOut} unit="ms" />
            <Metric label="上行待发（bufferedAmount）" value={metrics.upBuf} unit="ms" good={metrics.upBuf != null && metrics.upBuf <= 300} />
            <Metric label="单块计算均值" value={metrics.computeMean} unit="ms" />
            <Metric label="单块计算 P95" value={metrics.computeP95} unit="ms" />
            <Metric label="单块计算最大值" value={metrics.computeMax} unit="ms" />
            <Metric label="RTF（客户端累计）" value={metrics.rtf} unit="" d={3} />
            <Metric label="播放抖动缓冲（中位）" value={metrics.queue} unit="ms" />
            <Metric label="已处理" value={metrics.audioS} unit="s" />
          </div>
          <p className="hint">
            端到端估计 = 算法延迟 + 服务端输入积压 + 上行待发 + RTT/2 + 播放缓冲；算法延迟为“已收样本 − 已发输出”的差值，含 32ms 分块与 STFT 前置量。
            丢帧：输入 {fmt(metrics.dropIn)} ms / 发送 {fmt(metrics.dropOut)} ms / 本地 {fmt(metrics.clientDrop)} ms；旁路降级 {fmt(metrics.skipped)} ms；超预算块 {metrics.overruns ?? 0} 次；播放欠载 {metrics.underruns ?? 0} 次。
          </p>
          {summary && (
            <div className="summary">
              <div className="kv">
                <span>块数</span>
                <b>{summary.chunks}</b>
                <span>音频时长</span>
                <b>{summary.audio_s}s</b>
                <span>计算总时长</span>
                <b>{summary.compute_total_ms}ms</b>
                <b>均值</b>
                <b>{summary.compute_mean_ms}ms</b>
                <span>P95</span>
                <b>{summary.compute_p95_ms}ms</b>
                <span>RTF</span>
                <b>{summary.rtf}</b>
              </div>
            </div>
          )}
          <p className="foot">
            模型：{health?.model || "x16_tfmap_causal_bsrnn"} · 导出校验{" "}
            {health?.export?.pass ? "通过" : "未知"} · ONNX 与 PyTorch 最大误差{" "}
            {health?.export ? health.export.step_max_abs_diff_est.toExponential(2) : "–"}
          </p>
        </section>
      </div>
      </main>
    </div>
  );
}

function Metric({ label, value, unit, good, d = 1 }) {
  return (
    <div className={`metric ${good === true ? "good" : good === false ? "warn" : ""}`}>
      <span>{label}</span>
      <b>
        {fmt(value, d)}
        <i>{unit}</i>
      </b>
    </div>
  );
}

function phaseLabel(p) {
  return {
    idle: "待机",
    loading: "载入中",
    "enroll-recording": "录制声纹",
    enrolling: "声纹编码",
    streaming: "流式推理中",
    stopping: "收尾",
    done: "已完成",
    error: "错误",
  }[p] || p;
}
