# Fraud Detection & AML Copilot

Banka işlemlerinde dolandırıcılığı (fraud) tespit eden, her şüpheli işlem için sade Türkçe bir "Şüpheli İşlem Raporu" taslağı üreten ve tüm bunu risk-bazlı karar veren bir **LangGraph ajanına** bağlayan uçtan uca bir sistem. Böylece bir uyum (AML) müfettişi hem tespiti, hem gerekçesini, hem de otomatik bir kararı inceleyip onaylayabileceği bir formatta alır.

*An end-to-end system that flags fraudulent transactions, drafts a plain-Turkish suspicious-transaction report for each, and wires it all into a risk-routing LangGraph agent — so a compliance (AML) investigator gets the detection, the reasoning, and an automated decision in one reviewable flow.*

## Neden bu repo var / Why this repo exists

Sadece "bu işlem şüpheli" diyen bir model işin yarısını yapar. Bir bankada müfettişin hâlâ *neden* şüpheli olduğunu anlaması, bir rapor yazması ve ne yapılacağına karar vermesi gerekir — yavaş, manuel bir iş. Bu proje tespiti bir açıklama ve karar katmanıyla birleştirir: **XGBoost** şüpheliyi bulur, **SHAP** nedenini açıklar, **LLM** Türkçe rapor taslağı yazar, ve bir **LangGraph ajanı** risk seviyesine göre kararı yönlendirir.

Veri, herkese açık [PaySim](https://www.kaggle.com/datasets/ntnu-testimon/paysim1) sentetik mobil ödeme veri setidir. Gerçek müşteri verisi kullanılmamıştır.

*A model that just says "this is fraud" does half the job — an investigator still has to understand why, write a report, and decide what to do. This pairs detection with explanation and decision-making: XGBoost finds it, SHAP explains it, an LLM drafts the Turkish report, and a LangGraph agent routes the decision by risk level. Data is the public PaySim synthetic dataset; no real customer data.*

## Ne yapıyor / What it does

### 1. Veri keşfi / Data exploration
Fraud çok nadir (~%0.13), yani bu bir dengesiz sınıflandırma problemi — bu hem model seçimini hem metrikleri belirler. Önemli bir bulgu: fraud **yalnızca TRANSFER ve CASH_OUT** işlemlerinde görülüyor, ödeme veya para girişinde değil. Mantıklı (dolandırıcı parayı *çıkarır*) ve modeli bu iki tipe odaklamamızı sağlıyor.

*Fraud is rare (~0.13%) → imbalanced problem. Key finding: fraud occurs only in TRANSFER and CASH_OUT, which focuses the model.*

### 2. Özellik türetme / Feature engineering
Ham sütunlar tek başına yetmez. Normal bir işlemde `eski bakiye − yeni bakiye = tutar` olmalı; tutmuyorsa bir tuhaflık var. İki türetilmiş özellik bunu yakalar: `errorBalanceOrig` (gönderen tutarsızlığı) ve `errorBalanceDest` (alıcı tutarsızlığı). Fraud işlemlerde hesap sıfırlanır, bu özellikler tam onu yakalar. ID sütunları atılır (leakage riski), işlem tipi ikili kodlanır.

*Derived balance-inconsistency features catch the account being emptied; ID columns dropped (leakage); type binary-encoded.*

### 3. Tespit / Detection (XGBoost)
XGBoost tablo verisi için güçlü — doğrusal bir modelin kaçıracağı ilişkileri yakalar. Dengesizlik `scale_pos_weight` ile, bölme `stratify` ile ele alınır. Accuracy bilerek kullanılmaz ("hepsi normal" diyen model %99.9 alır, tek fraud yakalamaz).

| Metrik / Metric   | Skor / Score |
| ----------------- | ------------ |
| PR-AUC            | 0.997        |
| ROC-AUC           | 0.998        |
| Recall (fraud)    | 0.997        |
| Precision (fraud) | 0.981        |

Recall 0.997 → gerçek fraud'ların ~%99.7'si yakalanıyor; precision 0.981 → yanlış alarm düşük.

### 4. Açıklanabilirlik / Explainability (SHAP)
SHAP her tahmini özellik katkılarına ayırır. İki amaç: modelin tek bir sızıntılı özelliğe değil birden çok sinyale dayandığını doğrulamak, ve tek bir işlem için "neden" faktörlerini çıkarmak. En güçlü sinyal, türetilen `errorBalanceOrig` (bakiye tutarsızlığı), ardından işlem tutarı ve hesabın sıfırlanması.

*SHAP validates the model uses several signals (not one leaky feature) and extracts the "why". Strongest signal: the derived errorBalanceOrig.*

### 5. LLM ile Türkçe rapor / Turkish report via LLM
En yüksek riskli işlem için SHAP faktörleri ve işlem detayları bir LLM'e (Groq üzerinden Llama-3.3) verilir; model teknik terim içermeyen, resmi Türkçe bir Şüpheli İşlem Raporu taslağı üretir.

*The SHAP factors and transaction details are passed to an LLM (Llama-3.3 via Groq), which drafts a formal Turkish suspicious-transaction report with no technical jargon.*

### 6. LangGraph ajanı / Agent workflow
Tespit, açıklama ve raporlama adımları bir **LangGraph state machine**'e bağlanır. Ajan risk-bazlı koşullu karar verir:

- **Fraud olasılığı yüksekse** (> 0.80) → rapor üretilir, işlem müfettişe işaretlenir (`REPORTED`)
- **Düşükse** → otomatik onaylanır, rapor üretilmez (`AUTO_APPROVED`)

Bu, gerçek bir AML sürecinin mantığıdır: her işleme rapor yazılmaz; sadece riskli olanlar bir insana (human-in-the-loop) gider. Böylece sistem sadece tahmin eden bir model değil, karar veren bir iş akışıdır.

*The steps are wired into a LangGraph state machine with risk-based routing: high-risk transactions get a report and are flagged for a human investigator; low-risk ones are auto-approved. This makes it an operational, human-in-the-loop workflow — not just a model.*

**Örnek / Example (fetch → analyze → route → report):**
> **ŞÜPHELİ İŞLEM RAPORU — Karar: REPORTED**
> İşlem Tipi: TRANSFER · Tutar: 7.703.574,71 · Fraud olasılığı: %100
> Risk faktörleri: errorBalanceOrig (+9.82), amount (+2.32), newbalanceDest (+1.19)
> Gönderen bakiyenin aniden sıfıra düşmesi ve yüksek tutar nedeniyle işlem şüpheli bulunmuş, detaylı inceleme ve ilgili mercilere bildirim önerilmiştir.

Bu zincir — **model (bayrak) → SHAP (neden) → LLM (rapor) → ajan (karar)** — projenin amacıdır.

## Kurulum / Setup

Google Colab için hazırlandı (GPU gerekmez; LLM adımı hosted API kullanır).

1. PaySim CSV'yi [Kaggle](https://www.kaggle.com/datasets/ntnu-testimon/paysim1)'dan indirip notebook'a yükle.
2. Colab **Secrets**'a `GROQ_API_KEY` ekle.
3. `Runtime → Run all`.

```
pip install pandas numpy scikit-learn xgboost shap groq langgraph
```

## Teknik yığın / Tech stack
scikit-learn · XGBoost · SHAP (TreeExplainer) · Groq / Llama-3.3 · LangGraph · pandas / NumPy

## Bilinen sınırlar / Known limitations
- **Sentetik veri / Synthetic data:** PaySim gerçek değil; gerçek fraud desenleri daha karmaşıktır.
- **Türetilen bakiye özellikleri güçlü sinyal veriyor** — yüksek skoru kısmen açıklar; gerçekte tek bir özellik bu kadar belirleyici olmaz. Bu yaklaşım (TRANSFER/CASH_OUT odağı + bakiye-hata özellikleri) PaySim için bilinen analiz yöntemini temel alır; projenin katkısı üstteki SHAP + LLM + ajan katmanıdır.
- **Ajan risk eşiği (0.80) sabittir** — gerçek sistemde iş kuralları ve maliyet/fayda dengesine göre ayarlanır.
- **LLM raporu taslaktır, resmi belge değildir** — müfettiş incelemesi gerekir.
- **Tek train/test split**, hiperparametre optimizasyonu yok — çalışan prototip, tune edilmiş production modeli değil.
