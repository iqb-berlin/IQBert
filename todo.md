# IQBERT

## Technisch
* Feinjustierbares Bertmodell als ICS Service verpacken, damit er über die ICS Oberfläche ansprechbar ist
* IQBert ICS service deployen

## Technisch-Inhaltlich
* verschiedene trainingsdaten 
* relevante trainings-parameter identifizieren (z. B. weight_decay usw.) und konfigurierbar machen
* andere Basisimages als bert-base-uncased probieren - eventuell konfigurierbar machen
* weiteres Feinjustieren von bereits Feinjustierten modellen
* learning rate scheduler ausprobieren
* Validierung während des lernvorgangs und automatische Auswahl des besten Zustands
* Auf mehr codes als nur 0 und 1 hin trainieren
* verschiedene Tokenizer vorschalten (konfigurierbar machen?)
* Anstatt BertForSequenceClassification z. B. BertForQuestionAnswering probieren 