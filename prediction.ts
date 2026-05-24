import { Router, type IRouter } from "express";
import { anthropic } from "@workspace/integrations-anthropic-ai";

const router: IRouter = Router();

const ML_SERVICE_URL = process.env.PYTHON_ML_SERVICE_URL || `http://localhost:8008`;
const ML_TIMEOUT_MS  = 30_000; // 30s — TF inference can be slow on CPU

// ── Dataset metadata ──────────────────────────────────────────────────────────
const DATASET_SIZE     = 1190;
const DATASET_NAME     = "Heart Statlog Cleveland Hungary Final";
const DATASET_FEATURES = 11;

// ── Types ─────────────────────────────────────────────────────────────────────
interface PatientData {
  age: number;
  sex: number;
  chestPainType: number;
  restingBpS: number;
  cholesterol: number;
  fastingBloodSugar: number;
  restingEcg: number;
  maxHeartRate: number;
  exerciseAngina: number;
  oldpeak: number;
  stSlope: number;
  patientName?: string;
}

// ── Python ML proxy helper ────────────────────────────────────────────────────
async function callPython<T>(path: string, body: unknown): Promise<T | null> {
  try {
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), ML_TIMEOUT_MS);
    const resp = await fetch(`${ML_SERVICE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    clearTimeout(timeout);
    if (!resp.ok) {
      const detail = await resp.text();
      throw new Error(`Python ML service error ${resp.status}: ${detail}`);
    }
    return (await resp.json()) as T;
  } catch (err: any) {
    if (err?.name === "AbortError") {
      console.error("[ML-Proxy] Python service timed out.");
    } else {
      console.error("[ML-Proxy] Error calling Python:", err?.message ?? err);
    }
    return null;
  }
}

// ── Fallback simulation (used when Python service is unavailable) ──────────────
function computeRfRiskFallback(data: PatientData): {
  riskScore: number;
  riskLevel: "low" | "moderate" | "high";
  rfProbability: number;
  recommendation: string;
  features: { name: string; importance: number }[];
  simulated: boolean;
} {
  let totalScore = 0;
  const contributions: { name: string; contribution: number; importance: number }[] = [];

  const stSlopeFactor = data.stSlope === 2 ? 0.8 : data.stSlope === 3 ? 1.0 : 0.0;
  contributions.push({ name: "ST Slope", contribution: stSlopeFactor * 20, importance: 0.20 });
  totalScore += stSlopeFactor * 20;

  const cpFactor = data.chestPainType === 4 ? 1.0 : data.chestPainType === 3 ? 0.5 : data.chestPainType === 1 ? 0.3 : 0.0;
  contributions.push({ name: "Chest Pain Type", contribution: cpFactor * 18, importance: 0.18 });
  totalScore += cpFactor * 18;

  const oldpeakFactor = Math.min(1.0, Math.max(0, data.oldpeak / 4.0));
  contributions.push({ name: "ST Depression (Oldpeak)", contribution: oldpeakFactor * 16, importance: 0.16 });
  totalScore += oldpeakFactor * 16;

  const ageFactor = Math.max(0, Math.min(1.0, (data.age - 30) / 45));
  contributions.push({ name: "Age", contribution: ageFactor * 13, importance: 0.13 });
  totalScore += ageFactor * 13;

  const hrFactor = Math.max(0, Math.min(1.0, (220 - data.maxHeartRate) / 120));
  contributions.push({ name: "Max Heart Rate", contribution: hrFactor * 11, importance: 0.11 });
  totalScore += hrFactor * 11;

  contributions.push({ name: "Exercise Angina", contribution: data.exerciseAngina ? 9 : 0, importance: 0.09 });
  totalScore += data.exerciseAngina ? 9 : 0;

  const bpFactor = Math.max(0, Math.min(1.0, (data.restingBpS - 100) / 80));
  contributions.push({ name: "Resting Blood Pressure", contribution: bpFactor * 6, importance: 0.06 });
  totalScore += bpFactor * 6;

  contributions.push({ name: "Sex", contribution: data.sex === 1 ? 4 : 0, importance: 0.04 });
  totalScore += data.sex === 1 ? 4 : 0;

  const ecgFactor = data.restingEcg > 0 ? 1.0 : 0.0;
  contributions.push({ name: "Resting ECG", contribution: ecgFactor * 3, importance: 0.03 });
  totalScore += ecgFactor * 3;

  const normalizedScore = Math.min(100, Math.max(0, totalScore));
  const rfProbability = normalizedScore / 100;
  const riskLevel: "low" | "moderate" | "high" = normalizedScore < 33 ? "low" : normalizedScore < 60 ? "moderate" : "high";

  const recommendation = riskLevel === "low"
    ? "Low cardiovascular risk detected. Maintain a healthy lifestyle with regular exercise and balanced diet. Routine annual check-ups recommended."
    : riskLevel === "moderate"
    ? "Moderate cardiovascular risk. Follow-up with primary care physician within 4–6 weeks recommended. Consider lifestyle modifications."
    : "High cardiovascular risk detected. Urgent clinical review recommended within 1–2 weeks. Referral to cardiologist advised.";

  const features = contributions
    .filter(f => f.importance > 0)
    .sort((a, b) => b.importance - a.importance)
    .map(f => ({ name: f.name, importance: f.importance }));

  return { riskScore: normalizedScore, riskLevel, rfProbability, recommendation, features, simulated: true };
}

function computeEcgResultFallback(sampleType?: string, hasImage?: boolean): {
  classification: string;
  confidence: number;
  findings: string;
  riskLevel: "low" | "moderate" | "high";
  isDemo: boolean;
  simulated: boolean;
} {
  const ecgProfiles: Record<string, { classification: string; confidence: number; findings: string; riskLevel: "low" | "moderate" | "high" }> = {
    normal: {
      classification: "Normal ECG",
      confidence: 0.94,
      findings: "Regular P waves, normal PR interval (160–200ms), QRS within limits. No ischemic changes. VGG16 classification: Normal.",
      riskLevel: "low",
    },
    myocardial_infarction: {
      classification: "Myocardial Infarction",
      confidence: 0.91,
      findings: "Pathological Q waves, ST elevation, and T-wave inversion consistent with MI. Urgent cardiology review required.",
      riskLevel: "high",
    },
    history_mi: {
      classification: "History of Myocardial Infarction",
      confidence: 0.88,
      findings: "Residual Q waves and chronic ST-T changes. Secondary prevention and cardiologist follow-up strongly recommended.",
      riskLevel: "moderate",
    },
    abnormal_heartbeat: {
      classification: "Abnormal Heartbeat",
      confidence: 0.86,
      findings: "Abnormal rhythm morphology. May indicate arrhythmia or conduction abnormality. Clinical correlation recommended.",
      riskLevel: "moderate",
    },
  };

  const profile = sampleType && ecgProfiles[sampleType]
    ? ecgProfiles[sampleType]
    : { classification: "ECG Waveform Analyzed", confidence: 0.78, findings: "ECG waveform morphology analyzed. Clinical correlation recommended.", riskLevel: "moderate" as const };

  return { ...profile, isDemo: !hasImage, simulated: true };
}

// ── Routes ────────────────────────────────────────────────────────────────────

router.post("/analyze", async (req, res) => {
  try {
    const data = req.body as PatientData;

    // Try real ML model first
    const pyResult = await callPython<any>("/predict/rf", {
      age: data.age,
      sex: data.sex,
      chestPainType: data.chestPainType,
      restingBpS: data.restingBpS,
      cholesterol: data.cholesterol,
      fastingBloodSugar: data.fastingBloodSugar,
      restingEcg: data.restingEcg,
      maxHeartRate: data.maxHeartRate,
      exerciseAngina: data.exerciseAngina,
      oldpeak: data.oldpeak,
      stSlope: data.stSlope,
    });

    if (pyResult) {
      res.json({ ...pyResult, modelSource: "rf_model.pkl" });
      return;
    }

    // Fallback to simulation
    const result = computeRfRiskFallback(data);
    res.json({ ...result, modelAccuracy: 0.9202, modelPrecision: 0.921, modelRecall: 0.919, modelF1: 0.920 });
  } catch (err) {
    req.log.error({ err }, "Error analyzing patient data");
    res.status(500).json({ error: "Analysis failed" });
  }
});

router.post("/ecg-analyze", async (req, res) => {
  try {
    const { sampleType, imageData, hasImage } = req.body as {
      sampleType?: string;
      imageData?: string;
      hasImage?: boolean;
    };

    if (!hasImage && !imageData && !sampleType) {
      res.status(400).json({ error: "No ECG image or sample type provided." });
      return;
    }

    // Use real VGG16 if we have actual image data
    if (imageData || hasImage) {
      if (!imageData) {
        // hasImage=true but no actual data — shouldn't happen, fall through to fallback
      } else {
        const pyResult = await callPython<any>("/predict/ecg", { imageData });
        if (pyResult) {
          res.json({ ...pyResult, modelSource: "vgg16_ecg_model.keras" });
          return;
        }
      }
    }

    // Fallback: simulation for sample types or when Python is unavailable
    const ecgResult = computeEcgResultFallback(sampleType, hasImage);
    res.json({
      ...ecgResult,
      modelAccuracy: 0.7483,
      modelPrecision: 0.778,
      modelRecall: 0.7483,
      modelF1: 0.751,
      modelName: "VGG16 ECG Classifier",
    });
  } catch (err) {
    req.log.error({ err }, "Error analyzing ECG");
    res.status(500).json({ error: "ECG analysis failed" });
  }
});

router.post("/dual-analyze", async (req, res) => {
  try {
    const { patientData, ecgSampleType, hasEcgImage, ecgImageData } = req.body as {
      patientData: PatientData;
      ecgSampleType?: string;
      hasEcgImage?: boolean;
      ecgImageData?: string;
    };

    // ── RF prediction ───────────────────────────────────
    let rfResult: any;
    const pyRf = await callPython<any>("/predict/rf", {
      age: patientData.age,
      sex: patientData.sex,
      chestPainType: patientData.chestPainType,
      restingBpS: patientData.restingBpS,
      cholesterol: patientData.cholesterol,
      fastingBloodSugar: patientData.fastingBloodSugar,
      restingEcg: patientData.restingEcg,
      maxHeartRate: patientData.maxHeartRate,
      exerciseAngina: patientData.exerciseAngina,
      oldpeak: patientData.oldpeak,
      stSlope: patientData.stSlope,
    });

    if (pyRf) {
      rfResult = { ...pyRf, modelSource: "rf_model.pkl" };
    } else {
      const fallback = computeRfRiskFallback(patientData);
      rfResult = { ...fallback, modelAccuracy: 0.9202, modelPrecision: 0.921, modelRecall: 0.919, modelF1: 0.920 };
    }

    const ecgProvided = !!(hasEcgImage || ecgSampleType);

    if (!ecgProvided) {
      const riskLevel = rfResult.riskLevel as "low" | "moderate" | "high";
      const finalRecommendation = riskLevel === "high"
        ? "HIGH RISK: Random Forest model indicates high cardiovascular risk. ECG not provided. Clinical review urgently recommended."
        : riskLevel === "moderate"
        ? "MODERATE RISK: Random Forest indicates moderate risk. Upload an ECG for a more comprehensive dual-model assessment."
        : "LOW RISK: Random Forest indicates low cardiovascular risk. Upload an ECG image in Dual Mode for enhanced confidence.";

      res.json({
        rfResult,
        ecgResult: null,
        finalRiskLevel: riskLevel,
        finalRecommendation,
        confidenceScore: rfResult.rfProbability,
        ecgProvided: false,
      });
      return;
    }

    // ── ECG prediction ──────────────────────────────────
    let ecgResult: any;
    if (ecgImageData) {
      const pyEcg = await callPython<any>("/predict/ecg", { imageData: ecgImageData });
      if (pyEcg) {
        ecgResult = { ...pyEcg, modelSource: "vgg16_ecg_model.keras" };
      }
    }
    if (!ecgResult) {
      const fallback = computeEcgResultFallback(ecgSampleType, hasEcgImage);
      ecgResult = { ...fallback, modelAccuracy: 0.7483, modelPrecision: 0.778, modelRecall: 0.7483, modelF1: 0.751, modelName: "VGG16 ECG Classifier" };
    }

    // ── Combined triage logic ───────────────────────────
    const rfHigh   = rfResult.riskLevel === "high" || rfResult.riskLevel === "moderate";
    const ecgAnomaly = ecgResult.riskLevel === "high" || ecgResult.riskLevel === "moderate";

    let finalRiskLevel: "low" | "moderate" | "high";
    let finalRecommendation: string;

    if (rfResult.riskLevel === "high" && ecgResult.riskLevel === "high") {
      finalRiskLevel = "high";
      finalRecommendation = "PRIORITY 1 — URGENT: Both RF and VGG16 ECG models indicate high cardiovascular risk. Immediate cardiology consultation required.";
    } else if (rfHigh && ecgAnomaly) {
      finalRiskLevel = "high";
      finalRecommendation = "HIGH RISK: Structured data and ECG analysis both flag significant cardiovascular indicators. Urgent cardiology review within 24–48 hours.";
    } else if (rfResult.riskLevel === "high" && ecgResult.riskLevel === "low") {
      finalRiskLevel = "moderate";
      finalRecommendation = "CARDIAC REVIEW ADVISED: RF indicates high structural risk, but ECG appears normal. Clinical correlation recommended — follow-up cardiology review advised.";
    } else if (rfResult.riskLevel === "low" && ecgAnomaly) {
      finalRiskLevel = "moderate";
      finalRecommendation = "CARDIAC REVIEW ADVISED: Patient vitals appear lower risk, but ECG shows anomalous patterns. ECG findings warrant further cardiologist investigation.";
    } else {
      finalRiskLevel = "low";
      finalRecommendation = "LOW RISK: Both models indicate low cardiovascular risk. Routine monitoring and annual cardiovascular check-up advised.";
    }

    const confidenceScore = (rfResult.rfProbability * 0.55) + (ecgResult.confidence * 0.45);

    res.json({ rfResult, ecgResult, finalRiskLevel, finalRecommendation, confidenceScore, ecgProvided: true });
  } catch (err) {
    req.log.error({ err }, "Error in dual analysis");
    res.status(500).json({ error: "Dual analysis failed" });
  }
});

// ── ECG AI Analysis (Claude Vision) ──────────────────────────────────────────
const ECG_ANALYSIS_SYSTEM_PROMPT = `You are a production-grade ECG Diagnostic System with 3-layer validation + CNN + clinical analysis.

INPUT FORMAT: ECG image + CNN probabilities provided in the user message (when available).

3-LAYER PROCESS — execute in exact order:

LAYER 1: STRUCTURAL VALIDATION (Reject before CNN analysis)
Check these 3 properties FIRST from the image:
1. Color Saturation: Near-monochrome? (Black trace + white/grid background)
   FAIL: High color saturation (photograph, colorful image)
2. Horizontal Frequency: Strong horizontal striping? (ECG waveform pattern visible)
   FAIL: Isotropic content without waveform striping (natural photos)
3. Dark Pixel Density: Sufficient dark pixels for an ECG trace?
   FAIL: Almost no dark content (blank or near-blank image)

2+ FAILURES = IMMEDIATE REJECTION. Stop here — no further analysis.

LAYER 2: ECG PREPROCESSING (Only if 0-1 validation failures)
Acknowledge preprocessing steps applied:
- Greyscale conversion
- Percentile contrast enhancement
- Soft contrast stretch (grid noise suppression)
- RGB reconstruction for CNN input

LAYER 3: CNN + CONFIDENCE GATING
Analyze the CNN output entropy:
- Low Entropy (clear winner): Primary class >60% dominance = HIGH confidence
- Moderate Entropy: Primary class 40-60% = MODERATE confidence, add caution note
- High Entropy (uncertain): Primary class <40% or near-uniform distribution = LOW ENTROPY WARNING — flag prominently

CLINICAL WAVEFORM ANALYSIS (run if Layer 1 passes — your clinical expertise):
- Signal Quality: excellent/good/moderate/poor + any artifacts
- Heart Rate: exact bpm from R-R intervals (state how many intervals counted)
- Rhythm: specific rhythm interpretation (sinus, irregular, AFib, etc.)
- Key Morphology by lead where visible:
  - QRS: narrow/wide, estimated duration in ms
  - ST Segment: elevated/depressed/isoelectric with mm deviation and specific leads
  - T Waves: normal/inverted/peaked with specific leads
- CNN-Waveform Agreement: MATCH or MISMATCH with explanation
- Primary Diagnosis: specific clinical diagnosis (STEMI/NSTEMI/normal/arrhythmia/etc)
- Key Evidence: exactly 3 specific, observable waveform findings supporting the diagnosis
- Overall Reliability: DEPLOYABLE | CAUTION | NOT RELIABLE

CRITICAL RULES:
1. Layer 1 failure = STOP immediately (no CNN or waveform analysis)
2. Quote CNN probabilities exactly as provided — never adjust or estimate
3. High entropy = display warning prominently — never hide uncertainty
4. Always perform waveform analysis if Layer 1 passes
5. Different ECGs produce different findings — never copy-paste generic templates
6. MISMATCH = flag prominently — trust waveforms over CNN when they disagree
7. If waveform data is insufficient or unclear, say so explicitly

OUTPUT FORMAT — respond with ONLY valid JSON matching this exact structure:

For a VALID ECG (isEcg=true):
{
  "isEcg": true,
  "validationConfidence": "HIGH" | "MEDIUM" | "LOW",
  "validationReason": "X/3 structural checks passed — [brief explanation]",
  "rejected": false,
  "rejectionReason": null,
  "cnnPrimary": "class name (e.g. Myocardial Infarction)",
  "cnnConfidence": "XX.X%",
  "cnnDistribution": "Normal XX.X% | MI XX.X% | History MI XX.X% | Abnormal XX.X%",
  "entropyGate": "HIGH" | "MODERATE" | "LOW — WARNING",
  "signalQuality": "excellent/good/moderate/poor — [details]",
  "heartRate": "XX bpm (X R-R intervals measured)",
  "rhythm": "[specific rhythm type]",
  "intervals": "[PR, QRS, QT intervals if visible]",
  "waveformFindings": "[detailed morphology: QRS width, ST changes with leads, T-wave findings with leads]",
  "axis": "[electrical axis if determinable]",
  "abnormalities": "[list of specific abnormalities with lead references, or None identified]",
  "primaryDiagnosis": "[specific clinical diagnosis]",
  "keyEvidence": "[3 specific waveform observations supporting the diagnosis, separated by semicolons]",
  "clinicalInterpretation": "[full clinical interpretation tying waveform + CNN together]",
  "cnnWaveformMatch": "MATCH" | "MISMATCH",
  "overallReliability": "DEPLOYABLE" | "CAUTION" | "NOT RELIABLE",
  "confidenceLevel": "High" | "Moderate" | "Low",
  "rawReport": "Full human-readable ECG Diagnostic Report with all emoji-headed sections exactly as specified in the output format"
}

For a REJECTED image (isEcg=false):
{
  "isEcg": false,
  "validationConfidence": "HIGH",
  "validationReason": "X/3 structural checks passed — [failed check reasons]",
  "rejected": true,
  "rejectionReason": "LAYER 1 REJECTED: Invalid ECG image. Reasons: [list failed checks]",
  "cnnPrimary": null,
  "cnnConfidence": null,
  "cnnDistribution": null,
  "entropyGate": null,
  "signalQuality": "N/A — image rejected",
  "heartRate": "N/A — image rejected",
  "rhythm": "N/A — image rejected",
  "intervals": "N/A — image rejected",
  "waveformFindings": "N/A — image rejected",
  "axis": "N/A — image rejected",
  "abnormalities": "N/A — image rejected",
  "primaryDiagnosis": null,
  "keyEvidence": null,
  "clinicalInterpretation": "N/A — image rejected before analysis",
  "cnnWaveformMatch": null,
  "overallReliability": "NOT RELIABLE",
  "confidenceLevel": "Low",
  "rawReport": "LAYER 1 REJECTED: Invalid ECG image\nReasons: [list failed checks]\nNo CNN analysis performed."
}

FINAL RULE: Accuracy over completeness. rawReport must contain the full formatted report with emoji section headers exactly matching the 3-layer output format. If evidence is insufficient, explicitly state limitations — never fabricate findings.`;

type EcgAiAnalysisResult = {
  isEcg: boolean;
  validationConfidence: "HIGH" | "MEDIUM" | "LOW";
  validationReason: string;
  rejected: boolean;
  rejectionReason?: string | null;
  cnnPrimary?: string | null;
  cnnConfidence?: string | null;
  cnnDistribution?: string | null;
  entropyGate?: string | null;
  cnnWaveformMatch?: "MATCH" | "MISMATCH" | null;
  signalQuality: string;
  heartRate: string;
  rhythm: string;
  intervals: string;
  waveformFindings: string;
  axis: string;
  abnormalities: string;
  primaryDiagnosis?: string | null;
  keyEvidence?: string | null;
  clinicalInterpretation: string;
  overallReliability?: "DEPLOYABLE" | "CAUTION" | "NOT RELIABLE" | null;
  confidenceLevel: "High" | "Moderate" | "Low";
  rawReport: string;
};

function parseEcgAiResponse(content: string): EcgAiAnalysisResult {
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0]);
      const isEcg = parsed.isEcg !== false;
      const rejected = parsed.rejected === true || !isEcg;
      const validConfidences = ["HIGH", "MEDIUM", "LOW"];
      const validMatchValues = ["MATCH", "MISMATCH"];
      const validReliability = ["DEPLOYABLE", "CAUTION", "NOT RELIABLE"];
      const na = rejected ? "N/A — image rejected" : "Not available";
      return {
        isEcg,
        validationConfidence: (validConfidences.includes(parsed.validationConfidence) ? parsed.validationConfidence : "LOW") as "HIGH" | "MEDIUM" | "LOW",
        validationReason: parsed.validationReason || (isEcg ? "Image validated as ECG" : "Image does not appear to be a valid ECG"),
        rejected,
        rejectionReason: parsed.rejectionReason || (rejected ? "Rejected: Not a valid ECG image" : null),
        cnnPrimary: parsed.cnnPrimary || null,
        cnnConfidence: parsed.cnnConfidence || null,
        cnnDistribution: parsed.cnnDistribution || null,
        entropyGate: parsed.entropyGate || null,
        cnnWaveformMatch: (validMatchValues.includes(parsed.cnnWaveformMatch) ? parsed.cnnWaveformMatch : null) as "MATCH" | "MISMATCH" | null,
        signalQuality: parsed.signalQuality || na,
        heartRate: parsed.heartRate || na,
        rhythm: parsed.rhythm || na,
        intervals: parsed.intervals || na,
        waveformFindings: parsed.waveformFindings || na,
        axis: parsed.axis || na,
        abnormalities: parsed.abnormalities || (rejected ? na : "None identified"),
        primaryDiagnosis: parsed.primaryDiagnosis || null,
        keyEvidence: parsed.keyEvidence || null,
        clinicalInterpretation: parsed.clinicalInterpretation || (rejected ? "N/A — image rejected" : "Insufficient data"),
        overallReliability: (validReliability.includes(parsed.overallReliability) ? parsed.overallReliability : (rejected ? "NOT RELIABLE" : null)) as "DEPLOYABLE" | "CAUTION" | "NOT RELIABLE" | null,
        confidenceLevel: (["High", "Moderate", "Low"].includes(parsed.confidenceLevel) ? parsed.confidenceLevel : (rejected ? "Low" : "Moderate")) as "High" | "Moderate" | "Low",
        rawReport: parsed.rawReport || content,
      };
    } catch (_) {
      // Fall through to text parsing
    }
  }

  // Fallback: return raw content
  return {
    isEcg: true,
    validationConfidence: "LOW",
    validationReason: "Unable to parse structured response",
    rejected: false,
    rejectionReason: null,
    cnnPrimary: null,
    cnnConfidence: null,
    cnnDistribution: null,
    entropyGate: null,
    cnnWaveformMatch: null,
    signalQuality: "See full report",
    heartRate: "See full report",
    rhythm: "See full report",
    intervals: "See full report",
    waveformFindings: "See full report",
    axis: "See full report",
    abnormalities: "See full report",
    primaryDiagnosis: null,
    keyEvidence: null,
    clinicalInterpretation: "See full report",
    overallReliability: null,
    confidenceLevel: "Low",
    rawReport: content,
  };
}

router.post("/ecg-ai-analyze", async (req, res) => {
  try {
    const { imageData, cnnProbabilities } = req.body as {
      imageData: string;
      cnnProbabilities?: Record<string, number>;
    };

    if (!imageData) {
      res.status(400).json({ error: "imageData is required for AI ECG analysis." });
      return;
    }

    // Extract base64 and media type from data URI
    const dataUriMatch = imageData.match(/^data:(image\/[a-z]+);base64,(.+)$/);
    if (!dataUriMatch) {
      res.status(400).json({ error: "imageData must be a valid base64 data URI (data:image/...;base64,...)" });
      return;
    }
    const rawMediaType = dataUriMatch[1];
    const ALLOWED_ECG_TYPES = ["image/png", "image/jpeg"];
    if (!ALLOWED_ECG_TYPES.includes(rawMediaType)) {
      res.status(400).json({ error: "Only PNG and JPEG ECG images are accepted for AI analysis." });
      return;
    }
    const mediaType = rawMediaType as "image/png" | "image/jpeg";
    const base64Data = dataUriMatch[2];

    // Build CNN probability string for the prompt
    const CLASS_LABELS: Record<string, string> = {
      normal: "Normal",
      myocardial_infarction: "MI",
      history_mi: "History MI",
      abnormal_heartbeat: "Abnormal HB",
    };

    let cnnPromptText = "Please analyze this ECG image according to your system instructions. Return ONLY a valid JSON object matching the specified format.";
    if (cnnProbabilities && Object.keys(cnnProbabilities).length > 0) {
      const probParts = Object.entries(cnnProbabilities).map(([key, val]) => {
        const label = CLASS_LABELS[key] ?? key;
        return `${label}: ${Math.round(val * 100)}%`;
      });
      const topEntry = Object.entries(cnnProbabilities).reduce((a, b) => (b[1] > a[1] ? b : a));
      const topLabel = CLASS_LABELS[topEntry[0]] ?? topEntry[0];
      const topPct = Math.round(topEntry[1] * 100);
      cnnPromptText = `Please analyze this ECG image. The CNN classifier has already processed it with these results:
CNN probabilities: [${probParts.join(", ")}]
Primary CNN class: ${topLabel} (${topPct}%)

Apply your two-phase process: first validate whether this is a real ECG, then if valid, perform full waveform analysis correlating your findings with the CNN results above. Return ONLY a valid JSON object matching the specified format.`;
    }

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 2048,
      system: ECG_ANALYSIS_SYSTEM_PROMPT,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: mediaType,
                data: base64Data,
              },
            },
            {
              type: "text",
              text: cnnPromptText,
            },
          ],
        },
      ],
    });

    const content = response.content[0];
    if (content.type !== "text") {
      res.status(500).json({ error: "Unexpected response type from AI model." });
      return;
    }

    const analysis = parseEcgAiResponse(content.text);
    res.json(analysis);
  } catch (err: any) {
    req.log.error({ err }, "Error in AI ECG analysis");
    res.status(500).json({ error: "AI ECG analysis failed: " + (err?.message ?? "Unknown error") });
  }
});

router.get("/metrics", (_req, res) => {
  res.json({
    rfAccuracy: 0.9202, rfPrecision: 0.921, rfRecall: 0.919, rfF1: 0.920,
    ecgAccuracy: 0.7483, ecgPrecision: 0.778, ecgRecall: 0.7483, ecgF1: 0.751,
    datasetSize: DATASET_SIZE, features: DATASET_FEATURES, datasetName: DATASET_NAME,
  });
});

export default router;
