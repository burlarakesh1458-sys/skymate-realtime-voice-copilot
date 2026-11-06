// SkyMate web client: mic -> 16 kHz PCM16 -> WebSocket; plays streamed TTS; live latency bars.
(() => {
  const $ = (id) => document.getElementById(id);
  const TARGET_SR = 16000;
  const CHUNK_MS = 100;
  const BARGE_IN_RMS = 0.04; // mic level that counts as "user talking over the reply"

  let ws = null, audioCtx = null, micStream = null, workletNode = null;
  let micOn = false, pttHeld = false;
  let playHead = 0; const playing = new Set();
  let currentTurn = null, botEl = null, eouAt = null, firstAudioSeen = false;

  // ---------------------------------------------------------------- UI helpers
  const setStatus = (s) => { const el = $("status"); el.textContent = s; el.className = `pill ${s}`; };
  const addMsg = (cls, text) => { const d = document.createElement("div"); d.className = `msg ${cls}`; d.textContent = text;
    $("log").appendChild(d); $("log").scrollTop = 1e9; return d; };
  const b64ToBytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const bytesToB64 = (u8) => { let s = ""; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000)); return btoa(s); };
  const send = (obj) => ws && ws.readyState === 1 && ws.send(JSON.stringify(obj));

  // ---------------------------------------------------------------- playback
  function ensureCtx() { if (!audioCtx) audioCtx = new AudioContext(); if (audioCtx.state === "suspended") audioCtx.resume(); return audioCtx; }
  function playPcm16(bytes, sr) {
    const ctx = ensureCtx();
    const i16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
    const buf = ctx.createBuffer(1, i16.length, sr);
    const ch = buf.getChannelData(0); for (let i = 0; i < i16.length; i++) ch[i] = i16[i] / 32768;
    const src = ctx.createBufferSource(); src.buffer = buf; src.connect(ctx.destination);
    playHead = Math.max(playHead, ctx.currentTime + 0.02); src.start(playHead); playHead += buf.duration;
    playing.add(src); src.onended = () => { playing.delete(src); if (!playing.size) setStatus(micOn ? "listening" : "idle"); };
    setStatus("speaking");
  }
  function stopPlayback() { for (const s of playing) { try { s.stop(); } catch (_) {} } playing.clear(); playHead = 0; }

  // ---------------------------------------------------------------- mic capture (AudioWorklet, downsample to 16 kHz)
  const workletSrc = `
    class Cap extends AudioWorkletProcessor {
      constructor(o){ super(); this.ratio = sampleRate / ${TARGET_SR}; this.acc = []; this.pos = 0; this.n = ${TARGET_SR * CHUNK_MS / 1000}; }
      process(inputs){ const x = inputs[0][0]; if (!x) return true;
        for (; this.pos < x.length; this.pos += this.ratio) this.acc.push(x[Math.floor(this.pos)]);
        this.pos -= x.length;
        while (this.acc.length >= this.n) { const f = this.acc.splice(0, this.n); const i16 = new Int16Array(f.length); let e = 0;
          for (let i = 0; i < f.length; i++) { const v = Math.max(-1, Math.min(1, f[i])); i16[i] = v * 32767; e += v * v; }
          this.port.postMessage({ pcm: i16.buffer, rms: Math.sqrt(e / f.length) }, [i16.buffer]); }
        return true; } }
    registerProcessor("cap", Cap);`;

  async function startMic() {
    const ctx = ensureCtx();
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([workletSrc], { type: "application/javascript" })));
    const srcNode = ctx.createMediaStreamSource(micStream);
    workletNode = new AudioWorkletNode(ctx, "cap");
    workletNode.port.onmessage = ({ data }) => {
      $("level").style.width = `${Math.min(100, data.rms * 400)}%`;
      if (playing.size && data.rms > BARGE_IN_RMS) { stopPlayback(); send({ type: "interrupt", reason: "barge_in" }); addMsg("sys", "↯ barge-in: interrupted reply"); }
      const handsFree = $("handsfree").checked;
      if (handsFree || pttHeld) send({ type: "audio.chunk", data: bytesToB64(new Uint8Array(data.pcm)) });
    };
    srcNode.connect(workletNode);
    micOn = true; $("mic").textContent = "🎙 Stop mic"; $("ptt").disabled = $("handsfree").checked; setStatus("listening");
  }
  function stopMic() { micStream?.getTracks().forEach((t) => t.stop()); workletNode?.disconnect(); micOn = false;
    $("mic").textContent = "🎙 Start mic"; $("ptt").disabled = true; setStatus("idle"); }

  // ---------------------------------------------------------------- latency bars
  const SEGS = [["asr_ms", "--asr"], ["llm_ttft_ms", "--ttft"], ["first_segment_ms", "--seg"], ["tts_ttfb_ms", "--tts"], ["overhead_ms", "--ovh"], ["network_ms", "--net"]];
  let maxTotal = 1500;
  function addBar(turnId, b, degraded) {
    const total = (b.e2e_ms || 0) + (b.network_ms || 0); maxTotal = Math.max(maxTotal, total);
    const row = document.createElement("div"); row.className = "bar-row";
    const bar = document.createElement("div"); bar.className = "bar"; bar.dataset.total = total;
    for (const [k, c] of SEGS) { const v = b[k] || 0; if (!v) continue; const d = document.createElement("div");
      d.style.background = `var(${c})`; d.dataset.v = v; d.title = `${k}: ${v.toFixed(0)} ms`; bar.appendChild(d); }
    row.innerHTML = `<span>#${turnId.split("-").pop()}</span>`; row.appendChild(bar);
    const lab = document.createElement("span"); lab.textContent = b.e2e_ms ? `${total.toFixed(0)} ms` : "no audio";
    if (degraded?.length) { lab.className = "deg"; lab.title = degraded.join(", "); lab.textContent += " ⚠"; }
    row.appendChild(lab); $("bars").prepend(row);
    for (const r of $("bars").querySelectorAll(".bar")) for (const d of r.children) d.style.width = `${(100 * d.dataset.v) / maxTotal}%`;
  }

  // ---------------------------------------------------------------- protocol
  function onEvent(ev) {
    switch (ev.type) {
      case "session.started":
        $("providers").textContent = `ASR ${ev.providers.asr.join(" → ")} · LLM ${ev.providers.llm.join(" → ")} · TTS ${ev.providers.tts.join(" → ")}`; break;
      case "vad.speech_start": if (playing.size) stopPlayback(); setStatus("listening"); break;
      case "vad.end_of_utterance": currentTurn = ev.turn_id; eouAt ??= performance.now(); firstAudioSeen = false; setStatus("thinking"); break;
      case "transcript.partial": $("partial").textContent = ev.text; break;
      case "transcript.final": $("partial").textContent = ""; if (ev.text) addMsg("user", ev.text); currentTurn = ev.turn_id; botEl = null; setStatus("thinking"); break;
      case "llm.token": if (!botEl) botEl = addMsg("bot", ""); botEl.textContent += ev.token; $("log").scrollTop = 1e9; break;
      case "tts.audio":
        if (ev.turn_id !== currentTurn) break; // stale audio from a cancelled turn
        if (!firstAudioSeen && eouAt != null) { firstAudioSeen = true; send({ type: "metrics.client", turn_id: ev.turn_id, client_e2e_ms: performance.now() - eouAt }); }
        playPcm16(b64ToBytes(ev.data), ev.sample_rate); break;
      case "ack": addMsg("sys", `⏳ ${ev.text} (${ev.reason})`); break;
      case "error": addMsg("err", `⚠ ${ev.stage || ""} ${ev.code}: ${ev.message}`); break;
      case "metrics": addBar(ev.turn_id, ev.budget, ev.degraded); break;
      case "turn.done": eouAt = null; if (!playing.size) setStatus(micOn ? "listening" : "idle"); if (ev.status === "cancelled") addMsg("sys", "turn cancelled"); break;
    }
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => { send({ type: "session.start", sample_rate: TARGET_SR, label: "web", server_vad: $("handsfree").checked });
      for (const id of ["mic", "interrupt", "text", "send"]) $(id).disabled = false; $("connect").textContent = "Connected"; $("connect").disabled = true; };
    ws.onmessage = (m) => onEvent(JSON.parse(m.data));
    ws.onclose = () => { setStatus("error"); addMsg("err", "socket closed"); $("connect").disabled = false; $("connect").textContent = "Reconnect"; };
  }

  // ---------------------------------------------------------------- wiring
  $("connect").onclick = () => { ensureCtx(); connect(); };
  $("mic").onclick = () => (micOn ? stopMic() : startMic().catch((e) => addMsg("err", `mic: ${e.message}`)));
  $("handsfree").onchange = () => { $("ptt").disabled = !micOn || $("handsfree").checked; };
  const pttDown = () => { pttHeld = true; stopPlayback(); send({ type: "interrupt", reason: "push_to_talk" }); setStatus("listening"); };
  const pttUp = () => { if (!pttHeld) return; pttHeld = false; eouAt = performance.now(); send({ type: "vad.end_of_utterance", client_ts: Date.now() }); };
  $("ptt").onmousedown = pttDown; $("ptt").onmouseup = pttUp; $("ptt").onmouseleave = pttUp;
  $("ptt").ontouchstart = pttDown; $("ptt").ontouchend = pttUp;
  $("interrupt").onclick = () => { stopPlayback(); send({ type: "interrupt", reason: "button" }); };
  $("textform").onsubmit = (e) => { e.preventDefault(); const t = $("text").value.trim(); if (!t) return;
    stopPlayback(); eouAt = performance.now(); firstAudioSeen = false; send({ type: "text.input", text: t, client_ts: Date.now() }); $("text").value = ""; };
})();
