# Holzofen Machine Learning

In dieser Arbeit wird ein Machine Learning Modell entwickelt, welches den Nachfüllzeitpunkt eines Holzofens vorhersagt. Dazu werden die Daten eines Holzofens mit Wetterdaten kombiniert und in einem Machine Learning Modell verarbeitet.

## Ordnerstruktur

Es gibt folgende Ordner und Dateien

### data Ordner

Der data Ordner enthält alle notwendigen Daten. \
Im pictures Ordner sind alle in den Notebooks verwendete Bilder abgelegt.

### Jupiter Notebooks

Die Notebooks liegen alle im Hauptverzeichnis. Die Notebooks sind in der Reihenfolge auszuführen, wie sie nummeriert sind.

#### 00_domain_kowledge.ipynb

Dieses Notebook enthält eine Einleitung der Arbeit und die zum Verständnis notwendigen Informationen zum Domänenwissen. Hier sind auch Bilder und erklärende Grafiken zum Ofen zu finden.

#### 01_data_preparation.ipynb

Dieses Notebook bereitet die Daten für das Machine Learning Modell vor. Dazu werden die Daten aus dem data Ordner geladen und mit Wetterdaten kombiniert. Das Ergebnis wird in der Datei `data/merged.csv` gespeichert. Ist die Datei bereits vorhanden, ist das Ausführen dieses Notebooks nicht notwendig.

# Installation

Es gibt zwei Möglichkeiten, wie dieses Projekte installiert werden kann.

## Git

Wenn das Projekt über Git installiert wird, müssen zunächst alle Daten geladen werden.
Die Daten werden über git lfs verwaltet. Dazu muss zunächst git lfs installiert werden.

```bash
git lfs install
git lfs pull
```

Nun müssen die Daten entpackt werden.

```bash
unzip data/raw_data.zip -d data/
```

## Zip Datei

Alternativ kann das Projekt auch als Zip Datei heruntergeladen werden. In diesem Fall enthält das Projekt bereits die Daten und muss nicht entpackt werden.

