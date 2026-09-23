# llm-classifier

Modul Python untuk classifier berbasis **prompt-logprob** atas model yang
diserve oleh vLLM. Model dipakai sebagai *scorer*, bukan generator: untuk
setiap kandidat, model diminta menghasilkan satu token secara konseptual
(`Yes` atau `No`), dan yang dipakai adalah

```text
score = log P(Yes) - log P(No)
```

Skor tersebut lalu dipakai untuk `noul`, `choice`, dan `score`.

## Instalasi (editable)

```bash
pip install -e /home/sultan/apps/llm-classifier
```

Untuk pengembangan:

```bash
pip install -e "/home/sultan/apps/llm-classifier[test]"
pytest /home/sultan/apps/llm-classifier
```

## Struktur

```
src/llm_classifier/
├── __init__.py     # export publik
├── ask.py          # ask(): multi-question paralel (ThreadPoolExecutor)
├── classifier.py   # PromptLogprobClassifier
├── images.py       # encode image -> data URI base64 + SHA-256 cache
└── scoring.py      # sigmoid/softmax/confidence + calibration helper
```

## Konfigurasi

Semua konfigurasi dimiliki oleh aplikasi, bukan modul. Nilai mayor yang
memengaruhi hasil scoring wajib diisi saat membangun classifier; nilai minor
memiliki default:

```python
from llm_classifier import PromptLogprobClassifier, ask

classifier = PromptLogprobClassifier.from_openai(
    base_url="http://192.168.16.41:11000/v1",
    model="Qwen3.8-27B",
    api_key="sk-EMPTY",            # vLLM biasanya menerima key apa pun
    decision_temperature=3.0,      # temperature kalibrasi
    seed=42,
    prompt_logprobs=20,
    # minor (punya default): sampling_temperature=0.0,
    # image_detail="low", timeout=120
)
```

Parameter `ask()` minor pun punya default: `use_cache=True`,
`image_paths=None`, `max_workers=min(8, jumlah_question)`.

Bila ingin memakai client OpenAI yang sudah ada (misal untuk sharing
connection pool), langsung panggil constructor utama dengan argumen yang
sama.

## Contoh

```python
result = ask(
    classifier,
    state="A customer says: I was charged twice for one order.",
    questions={
        "duplicate_charge": {
            "type": "noul",
            "instructions": "Does this message describe a duplicate charge?",
        },
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this issue?",
            "criteria": {
                "billing": "Problems with charges, invoices, refunds or payments.",
                "technical": "Service unavailable or a software malfunction.",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "Assess the urgency using only the given evidence.",
            "criteria": ["Routine", "Urgent", "Critical"],
        },
    },
)
print(result["answers"])
```

### Multimodal

`image_paths` menerima **list** path image (satu image pun tetap list).
Image dibaca dari filesystem dan dikirim sebagai data URI base64:

```python
result = ask(
    classifier,
    state="Gunakan isi gambar sebagai bukti utama.",
    questions={
        "is_ref_found": {
            "type": "noul",
            "instructions": "Apakah nomor referensi `002 / SPK / W1 / 1 /2025` ditemukan?",
        },
    },
    image_paths=["output/page_003.jpg"],
    max_workers=1,  # concurrency rendah disarankan untuk jalur image
)
```

Untuk `choice` dan `score`, classifier meminta model menghasilkan satu token
kode opsi dalam satu generation. Kode internal memakai `0` sampai `9`, lalu
`A` sampai `Z`, lalu `a` sampai `z` (maksimal 62 opsi). `noul` tetap memakai
scoring `Yes`/`No`.

`prompt_logprobs` juga dipakai sebagai jumlah `top_logprobs` untuk direct
label. Label yang tidak dikembalikan vLLM diberi probabilitas `0`; label yang
tersedia dinormalisasi ulang dengan `decision_temperature`.

## Kalibrasi

Score logprob biasanya belum terkalibrasi. Gunakan data validasi yang tidak
dipakai untuk prompt tuning:

```python
from llm_classifier import fit_decision_temperature

# samples: list of (scores, target_index)
# untuk noul: ([0.0, yes_no_score], 0 atau 1)
# untuk choice/score: (semua skor kandidat, index label benar)
calibrated = fit_decision_temperature(validation_samples)
classifier.decision_temperature = calibrated
```

## Catatan / keterbatasan

- `noul` berasal dari log-odds token `Yes`/`No`; token tersebut harus
  masing-masing satu token pada tokenizer server.
- `choice` dan `score` memakai satu token label dengan `logprobs`/`top_logprobs`.
  Label yang hilang dari top-k diperlakukan sebagai probabilitas nol.
- Jalur text-only memakai `/tokenize` + `/v1/completions`; jalur image memakai
  `chat.completions` dengan satu generation token.
- `decision_temperature` adalah temperature kalibrasi. `sampling_temperature`
  mengatur generation vLLM dan default-nya `0.0` agar top-1 deterministik.
- Cache aplikasi (`_score_cache`, `_image_cache`) menjamin request identik
  mengembalikan hasil identik pada proses yang sama; `clear_cache()`
  mengosongkan keduanya.
- Untuk production, simpan model revision, prompt version, seed, sampling
  settings, serta calibration temperature.
