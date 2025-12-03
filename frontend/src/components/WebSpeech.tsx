import React, { useRef, useState } from "react";

/* =============================
   Audio / WS singletons
============================= */
let audioContext: AudioContext | null = null;
let processor: ScriptProcessorNode | null = null;
let source: MediaStreamAudioSourceNode | null = null;
let ws: WebSocket | null = null;

/* Float32 → PCM16 */
function floatToPCM16(input: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(input.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

/* =============================
   Types
============================= */
type TranslationItem = {
  text: string;
  provisional: boolean;
  striked: boolean;
};

type AsrItem = {
  text: string;
  striked: boolean;
};

export default function App() {
  type Mode = "auto" | "fixed" | "multilang";
const [mode, setMode] = useState<Mode>("auto");
const [fixedLang, setFixedLang] = useState<string>("zh-TW"); // default Chinese

  const [running, setRunning] = useState(false);

  // ASR
  const [finalLines, setFinalLines] = useState<AsrItem[]>([]);
  const [partial, setPartial] = useState("");

  // Translation
  const [translations, setTranslations] = useState<TranslationItem[]>([]);
  const [partialTranslation, setPartialTranslation] = useState("");

  // Language UI
  const [langText, setLangText] = useState<string | null>(null);
  const [langState, setLangState] =
    useState<"detecting" | "mismatch" | "final">("detecting");
  const [langFlash, setLangFlash] = useState(false);

  const provisionalLangRef = useRef<string | null>(null);

  // mic
  const [micLevel, setMicLevel] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () =>
    setTimeout(
      () =>
        scrollRef.current?.scrollTo({
          top: scrollRef.current.scrollHeight,
          behavior: "smooth",
        }),
      0
    );

  /* =============================
     ▶ Start
  ============================= */
  const start = async () => {
    if (running) return;

    let wsUrl ="wss://instant-translation-backend-c4fpb7arc0f8d5cv.eastus-1.azurewebsites.net/ws";
   
   if (mode === "auto") {
     wsUrl =
       "wss://instant-translation-backend-c4fpb7arc0f8d5cv.eastus-1.azurewebsites.net/ws";
   } else if (mode === "fixed") {
     wsUrl = `wss://instant-translation-backend-c4fpb7arc0f8d5cv.eastus-1.azurewebsites.net/ws/fixed?lang=${fixedLang}`;
   } else if (mode === "multilang") {
     wsUrl =
       "wss://instant-translation-backend-c4fpb7arc0f8d5cv.eastus-1.azurewebsites.net/ws/multilang";
   }

    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      // Language: provisional
      if (msg.type === "lang") {
        provisionalLangRef.current = msg.lang;
        setLangText("Detecting language…");
        setLangState("detecting");
        return;
      }

      // Language: lock
      if (msg.type === "lang_locked") {
        const finalLang = msg.lang;
        const provisional = provisionalLangRef.current;

        // mismatch
        if (provisional && provisional !== finalLang) {
          setLangText(`Language corrected → ${finalLang}`);
          setLangState("mismatch");
          setLangFlash(true);

          setTimeout(() => {
            setLangFlash(false);
            setLangText(`Final: ${finalLang}`);
            setLangState("final");
          }, 1200);
        } else {
          setLangText(`Final: ${finalLang}`);
          setLangState("final");
        }

        provisionalLangRef.current = finalLang;
        return;
      }

      /* ----- 全部 invalidate（回溯重翻時用） ----- */
      if (msg.type === "invalidate_all_asr") {
        setFinalLines((p) => p.map((l) => ({ ...l, striked: true })));
        return;
      }

      if (msg.type === "invalidate_all_translation") {
        setTranslations((p) => p.map((t) => ({ ...t, striked: true })));
        return;
      }

      /* ----- ASR ----- */
      if (msg.type === "partial") {
        setPartial(msg.text);
        scrollToBottom();
        return;
      }

      if (msg.type === "final") {
        setFinalLines((p) => [...p, { text: msg.text, striked: false }]);
        setPartial("");
        scrollToBottom();
        return;
      }

      if (msg.type === "invalidate_asr") {
        setFinalLines((p) =>
          p.map((l, i) =>
            i === p.length - 1 ? { ...l, striked: true } : l
          )
        );
        return;
      }

      /* ----- Translation ----- */
      if (msg.type === "mid_translate") {
        setPartialTranslation(msg.translated);
        return;
      }

      if (msg.type === "final_translate") {
        setTranslations((prev) => {
          // replace_last: true → 把最後一條翻譯替換掉
          if (msg.replace_last && prev.length > 0) {
            const p = [...prev];
            p[p.length - 1] = {
              text: msg.translated,
              provisional: false,
              striked: false,
            };
            return p;
          }

          // 一般 append
          return [
            ...prev,
            {
              text: msg.translated,
              provisional: msg.provisional ?? false,
              striked: false,
            },
          ];
        });

        setPartialTranslation("");
        scrollToBottom();
        return;
      }

      if (msg.type === "invalidate_translation") {
        setTranslations((p) =>
          p.map((t, i) =>
            i === p.length - 1 ? { ...t, striked: true } : t
          )
        );
        return;
      }
    };

    ws.onopen = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      audioContext = new AudioContext({ sampleRate: 16000 });
      await audioContext.resume();

      source = audioContext.createMediaStreamSource(stream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const ch = e.inputBuffer.getChannelData(0);
        let sum = 0;
        for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
        setMicLevel(Math.min(1, Math.sqrt(sum / ch.length) * 5));

        ws!.send(floatToPCM16(ch));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      // reset
      setFinalLines([]);
      setTranslations([]);
      setPartial("");
      setPartialTranslation("");
      setLangText(null);
      setLangState("detecting");
      setRunning(true);
    };
  };

  const stop = () => {
    processor?.disconnect();
    source?.disconnect();
    audioContext?.close();
    ws?.close();
    setRunning(false);
  };


  const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: "#efece6",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    fontFamily:
      "-apple-system, BlinkMacSystemFont, 'Inter', Segoe UI, Arial, sans-serif",
  },
  card: {
    width: 720,
    maxWidth: "100%",
    background: "#faf9f7",
    borderRadius: 24,
    padding: 32,
    boxShadow: "0 45px 90px rgba(0,0,0,0.2)",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 16,
    marginBottom: 16,
  },
  buttonRow: {
    display: "flex",
    gap: 14,
    marginBottom: 16,
  },
  buttonPrimary: {
    flex: 1,
    padding: 14,
    borderRadius: 18,
    background: "#8fa3a6",
    color: "#fff",
    border: "none",
    fontSize: 16,
    cursor: "pointer",
  },
  buttonSecondary: {
    flex: 1,
    padding: 14,
    borderRadius: 18,
    background: "#c1cbc4",
    border: "none",
    fontSize: 16,
    cursor: "pointer",
  },
  transcript: {
    height: 420,
    overflowY: "auto",
    background: "#f3f4f2",
    borderRadius: 16,
    padding: 16,
    fontSize: 15,
    lineHeight: 1.5,
  },

  /* language status */
  langBox: {
    padding: "8px 14px",
    fontSize: 15,
    fontWeight: 600,
    borderRadius: 12,
    transition: "all 0.35s ease",
    borderLeft: "6px solid transparent",
    whiteSpace: "nowrap",
  },
  langDetect: {
    background: "#e6edf3",
    color: "#4a5568",
    borderLeftColor: "#a0aec0",
  },
  langMismatch: {
    background: "#fff3e0",
    color: "#dd6b20",
    borderLeftColor: "#dd6b20",
    boxShadow: "0 0 0 4px rgba(221,107,32,0.25)",
  },
  langFinal: {
    background: "#e6fffa",
    color: "#276749",
    borderLeftColor: "#38a169",
  },
};



  /* =============================
     🎨 UI
============================= */
  return (
  <div style={styles.page}>
    <div style={styles.card}>
      {/* title + language status */}
      <div style={styles.header}>
        <h1>🎙 Instant Translation</h1>

        {langText && (
          <div
            style={{
              ...styles.langBox,
              ...(langState === "final"
                ? styles.langFinal
                : langState === "mismatch"
                ? styles.langMismatch
                : styles.langDetect),
              transform: langFlash ? "scale(1.06)" : "scale(1)",
            }}
          >
            {langText}
          </div>
        )}
      </div>

      {/* ===== Mode Selector ===== */}
      <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
        <button
          onClick={() => setMode("auto")}
          style={{
            padding: "8px 14px",
            borderRadius: 12,
            border: "none",
            cursor: "pointer",
            background: mode === "auto" ? "#8fa3a6" : "#ddd",
            color: mode === "auto" ? "#fff" : "#333",
          }}
        >
          Auto Detect
        </button>

        <button
          onClick={() => setMode("fixed")}
          style={{
            padding: "8px 14px",
            borderRadius: 12,
            border: "none",
            cursor: "pointer",
            background: mode === "fixed" ? "#8fa3a6" : "#ddd",
            color: mode === "fixed" ? "#fff" : "#333",
          }}
        >
          Fixed Language
        </button>

        <button
          onClick={() => setMode("multilang")}
          style={{
            padding: "8px 14px",
            borderRadius: 12,
            border: "none",
            cursor: "pointer",
            background: mode === "multilang" ? "#8fa3a6" : "#ddd",
            color: mode === "multilang" ? "#fff" : "#333",
          }}
        >
          Multi Language
        </button>

        {mode === "fixed" && (
          <select
            value={fixedLang}
            onChange={(e) => setFixedLang(e.target.value)}
            style={{
              marginLeft: 8,
              padding: "8px 12px",
              borderRadius: 10,
            }}
          >
            <option value="en">English</option>
            <option value="zh">Chinese (Mandarin)</option>
            <option value="ja">Japanese</option>
            <option value="ko">Korean</option>
            <option value="th">Thai</option>
            <option value="vi">Vietnamese</option>
            <option value="id">Indonesian</option>
            <option value="ms">Malay</option>
            <option value="hi">Hindi</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
            <option value="de">German</option>
            <option value="pt">Portuguese</option>
          </select>
        )}
      </div>

      {/* ===== Controls ===== */}
      <div style={styles.buttonRow}>
        <button
          style={styles.buttonPrimary}
          disabled={running}
          onClick={start}
        >
          {running ? "Listening…" : "Start"}
        </button>
        <button
          style={styles.buttonSecondary}
          disabled={!running}
          onClick={stop}
        >
          Stop
        </button>
      </div>

      {/* mic level */}
      <div style={{ marginBottom: 12 }}>
        <div
          style={{
            height: 6,
            borderRadius: 999,
            background: "#d4d7d3",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${Math.round(micLevel * 100)}%`,
              height: "100%",
              background: "#8fa3a6",
              transition: "width 0.05s linear",
            }}
          />
        </div>
      </div>

      {/* ===== Transcript ===== */}
      <div style={styles.transcript} ref={scrollRef}>
        {/* Original */}
        {finalLines.map((l, i) => (
          <div key={i} style={{ display: "flex", gap: 6 }}>
            <span>🗣</span>
            {l.striked ? (
              <span style={{ textDecoration: "line-through", opacity: 0.7 }}>
                {l.text}
              </span>
            ) : (
              <span>{l.text}</span>
            )}
          </div>
        ))}

        {/* Translation */}
        {translations.map((t, i) => (
          <div key={`t${i}`} style={{ display: "flex", gap: 6 }}>
            <span>🌍</span>
            {t.striked ? (
              <span style={{ textDecoration: "line-through", opacity: 0.7 }}>
                {t.text}
              </span>
            ) : (
              <span>{t.text}</span>
            )}
          </div>
        ))}

        {/* partial */}
        {partial && (
          <div style={{ display: "flex", gap: 6 }}>
            <span>🗣</span>
            <span>{partial}</span>
          </div>
        )}

        {partialTranslation && (
          <div style={{ display: "flex", gap: 6 }}>
            <span>🌍</span>
            <span>{partialTranslation}</span>
          </div>
        )}
      </div>
    </div>
  </div>
);
}
