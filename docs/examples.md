# Actual prediction examples

[Interactive viewer](https://voidful.github.io/barbet-structured-sft/#examples) ·
[Full raw records](../reports/predictions_20260919.json) ·
[Per-example review](../reports/prediction_review.json)

Eight inputs were fixed before inference on 2026-09-19. The first four
are the lowest test index in each named task family with JSON serialization
and a prompt no longer than 2,048 tokens. Correct reference text is part
of each structured request. The other four are new handwritten natural
instructions. Generation was greedy, capped at 1,024 / 256 tokens.

| Example | Observed result |
|---|---|
| ASR correction | Correctly replaces 數位千章 with 數位簽章; confidence 0.98 versus target 0.97. |
| TTS spoken form | Correctly returns G P U 使用率為百分之九十二點五。; full JSON exact. |
| OCR correction | Correctly replaces 貧料 with 資料; confidence 0.98 versus target 0.97. |
| Translation post-edit | Corrects January 20 to January 19; full JSON exact. |
| Natural ASR instruction | Repeats the request instead of returning the corrected sentence. |
| Natural TTS instruction | Alters the digit instruction and repeats `xdb` until the token cap. |
| Natural OCR instruction | Labels 茶葉 instead of returning the order number. |
| Natural translation | Repeats the English sentence instead of translating. |

The four structured edited texts match their references; whole JSON exact
is 2/4. These are demonstrations, not a new benchmark. `decision: FAIL`
in a structured answer labels an erroneous input candidate, not an inference
process failure. All original outputs are retained without manual repair.
