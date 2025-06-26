# IQBERT

## Technisch/Infrastruktur
* Feinjustierbares Bertmodell als [ICS Service](https://github.com/iqb-specifications/coding-service) verpacken,
  damit er über die [ICS Oberfläche](https://github.com/iqb-berlin/ics-frontend) ansprechbar ist.
* Auf eine eines IQB-Servers (z. B. lab.iqb-berlin.de) service ausrollen und einrichten.

## Technisch-Inhaltlich
* Relevante Trainingsparameter identifizieren (z. B. weight_decay usw.) und konfigurierbar machen
* Andere pretrained models als bert-base-uncased probieren und konfigurierbar machen:
  * Verschiedene Berts: https://huggingface.co/google-bert/models
  * Auch GPT Modelle wären wohl möglich: https://huggingface.co/openai-community
* weiteres Feinjustieren von bereits feinjustierten Modellen ermöglichen
* learning rate scheduler ausprobieren
* Validierung während des lernvorgangs aktivieren und automatische Auswahl des besten Zustands
* Auf mehr Codes als nur 0 und 1 hin trainieren. Missingtypen beachten.
* Tokenizer optimieren und verschiedene Tokenizer probieren und das konfigurierbar machen.
* Andere heads als BertForSequenceClassification z. B. BertForQuestionAnswering probieren (konfigurierbar machen?)
* predictions mit certainty ausgeben