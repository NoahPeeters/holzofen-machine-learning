# Holzofen Machine Learning

In dieser Arbeit wird ein Machine Learning Modell entwickelt, welches den Nachfüllzeitpunkt eines Holzofens vorhersagt. Dazu werden die Daten eines Holzofens mit Wetterdaten kombiniert und in einem Machine Learning Modell verarbeitet.

## Aufbau

Um die Arbeit nachvollziehbar und verständlich aufzubauen, wurde die Arbeit in mehrere Schritte unterteilt. Für jeden Schritt gibt es ein eigenes Jupyter Notebook.
Die Jupiter Notebooks bauen inhaltlich aufeinander aus:

### 00_domain_kowledge.ipynb

Dieses Notebook enthält eine Einleitung der Arbeit und die zum Verständnis notwendigen Informationen zum Domänenwissen. Hier sind auch Bilder und erklärende Grafiken zum Ofen zu finden.

### 01_data_preparation.ipynb

Dieses Notebook bereitet die Daten für alle weiteren Schritte vor. Dazu werden die Daten aus dem data Ordner geladen und mit Wetterdaten kombiniert. Das Ergebnis wird in der Datei `data/merged.pkl` gespeichert. Daher ist es notwendig, dieses Notebook auszuführen, bevor die weiteren Notebooks ausgeführt werden können.

### 02_exploratory_data_analysis.ipynb

Dieses Notebook enthält die explorative Datenanalyse. Hier werden die Daten untersucht und erste Erkenntnisse gewonnen.

### 03_training_preparation.ipynb

Dieses Notebook betrachtet die beiden Konzepte Cross Validation und Modell Evaluation.
Es wird untersucht, welche Verfahren für diese beiden Konzepte für diese Arbeit geeignet sind.

### 04_classic.ipynb

Dieses Notebook enthält die klassischen Machine Learning Modelle. Es werden verschiedene Modelle trainiert und evaluiert.

### 05_xgboost_hyperparameter_tuning.ipynb

Dieses Notebook enthält das Hyperparameter Tuning für das XGBoost Modell. Es werden verschiedene Hyperparameter Kombinationen getestet und evaluiert.
**Dieses Notebook ist noch nicht fertig.**

### 06_final.ipynb

Dieses Notebook enthält das finale Modell sowie eine abschließende Evaluation.
**Dieses Notebook ist noch nicht fertig.**

### Weitere Daten

Neben den Jupyter Notebooks gibt es noch weitere Dateien, die für die Arbeit notwendig sind.

Im `data` Ordner befinden sich die Rohdaten sowie die aufbereiteten Daten (sofern `01_data_preparation.ipynb` ausgeführt wurde).

Im `pictures` Ordner befinden sich alle in den Notebooks verwendeten Bilder.

Die Datei `helper.py` enthält Funktionen, die in den Notebooks verwendet werden.

Die Datei `requirements.txt` enthält alle notwendigen Python Pakete, die für die Ausführung der Notebooks notwendig sind.

## Einrichtung 

Bei der Einrichtung müssen folgende Schritte durchgeführt werden:

### Python Umgebung

Für die Ausführung der Notebooks wird eine Python Umgebung benötigt. Alle benötigten Pakete sind in der Datei `requirements.txt` aufgelistet. Die Python Umgebung kann mit folgendem Befehl eingerichtet werden:

```bash
pip install -r requirements.txt
```

### Daten

Je nachdem, wie das Projekt bezogen wurde, müssen die Daten noch vorbereitet werden.

1. Wenn der Ordner `data/raw` existiert, sind alle Daten bereits vorbereitet und es ist nichts weiter zu tun.
2. Wenn die Daten `data/rawzip` existieren, müssen diese noch entpackt werden. Dazu kann folgender Befehl verwendet werden:

```bash
unzip data/raw_data.zip -d data/
```

3. Wenn die Daten noch nicht vorhanden sind, müssen diese zunächst bezogen werden. Wurde das Projekt über Git geladen, werden die Daten über `git lfs` verwaltet. Dazu muss zunächst `git lfs` installiert und die Daten bezogen werden.

```bash
git lfs install
git lfs pull
```