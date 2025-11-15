import React, { useRef, useState } from "react";

const WS_URL = "ws://localhost:8000/ws_stream";

export default function InstantTranslation() {
  const [log, setLog] = useState<string>("(Transcriptions will appear here)");
  const [isRecording, setIsRecording] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const start = async () => {
    if (isRecording) return;
    setIsRecording(true);

    setLog((prev) => (prev === "(Transcriptions will appear here)" ? "" : prev));

    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);

        if (msg.type === "partial") {
          setLog((prev) => {
            const lines = prev ? prev.split("\n") : [];
            if (lines.length === 0) {
              return `🟡 [${msg.lang}] ${msg.text}`;
            }

            const last = lines[lines.length - 1];
            if (last.startsWith("🟡") || last.startsWith("🟢")) {
              lines[lines.length - 1] = `🟡 [${msg.lang}] ${msg.text}`;
            } else {
              lines.push(`🟡 [${msg.lang}] ${msg.text}`);
            }
            return lines.join("\n");
          });
        } else if (msg.type === "final") {
          setLog((prev) => {
            const lines = prev ? prev.split("\n") : [];
            const last = lines[lines.length - 1];

            if (last && (last.startsWith("🟡") || last.startsWith("🟢"))) {
              lines[lines.length - 1] = `🟢 [${msg.lang}] ${msg.text}`;
            } else {
              lines.push(`🟢 [${msg.lang}] ${msg.text}`);
            }

            if (msg.translation) {
              lines.push(`🌍 ${msg.translation}`);
            }

            return lines.join("\n");
          });
        }
      } catch {}
    };

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = new AudioContext({ sampleRate: 16000 });
    ctxRef.current = ctx;

    const source = ctx.createMediaStreamSource(stream);
    sourceRef.current = source;

    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      wsRef.current.send(floatTo16BitPCM(input));
    };

    source.connect(processor);
    processor.connect(ctx.destination);
  };

  const stop = () => {
    setIsRecording(false);

    setLog((prev) => (prev ? prev + "\n⏹ Recording stopped" : "⏹ Recording stopped"));

    try {
      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      ctxRef.current?.close();
    } catch {}

    try {
      wsRef.current?.send("END");
      wsRef.current?.close();
    } catch {}
  };

  const floatTo16BitPCM = (float32: Float32Array) => {
    const buf = new ArrayBuffer(float32.length * 2);
    const view = new DataView(buf);
    for (let i = 0; i < float32.length; i++) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buf;
  };

  return (
    <div
      style={{
        padding: 24,
        fontFamily: "Inter, sans-serif",
        background: "#EAE6DF", // Morandi beige
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        color: "#4A4A48",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 700,
          background: "#F4F1EC", // soft Morandi gray-white
          padding: 28,
          borderRadius: 14,
          boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
        }}
      >
        <h2 style={{ marginTop: 0, fontSize: "1.6rem", color: "#5A5853" }}>
           Instant Translation
        </h2>

        {!isRecording ? (
          <button
            onClick={start}
            style={{
              background: "#40739cff",
              border: "none",
              padding: "10px 18px",
              borderRadius: 10,
              color: "#fff",
              fontSize: "1rem",
              cursor: "pointer",
            }}
          >
            🎙 Start Recording
          </button>
        ) : (
          <button
            onClick={stop}
            style={{
              background: "#c06255ff",
              border: "none",
              padding: "10px 18px",
              borderRadius: 10,
              color: "#fff",
              fontSize: "1rem",
              cursor: "pointer",
            }}
          >
            ⏹ Stop Recording
          </button>
        )}

        <pre
          style={{
            marginTop: 18,
            whiteSpace: "pre-wrap",
            background: "#ebe2d3ff",
            padding: 16,
            borderRadius: 12,
            height: 420,
            overflowY: "auto",
            fontSize: "1.08rem",
            lineHeight: "1.6em",
            color: "#4A4A48",
            border: "1px solid #D8D3CC",
          }}
        >
          {log}
        </pre>
      </div>
    </div>
  );
}
