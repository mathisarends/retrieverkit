# Review der öffentlichen Retrieval-API

Stand: 5. August 2026

## Kurzfazit

Das Paket hat eine gute Ausgangsbasis: kleine Ports, wenige Datentypen,
asynchrone I/O-Grenzen und Komposition statt eines großen RAG-Frameworks.
`Retriever` als gemeinsamer Nenner für Index, Fusion und Reranking ist die
richtige Grundidee.

Für eine öffentliche, langfristig stabile API ist der Kern aktuell jedoch zu
schmal. Vor allem `retrieve(query, limit)` und `chunk(text) -> list[str]`
lassen zentrale Informationen nicht durch die Pipeline fließen. Anwendungen
werden deshalb eigene, backend-spezifische Nebenwege bauen. Sobald diese
existieren, kann der Core sie später kaum noch aufnehmen, ohne APIs zu brechen.

Vor einem stabilen Release sollten drei Dinge festgelegt werden:

1. ein erweiterbares Request-Modell für Suche und Filter,
2. durchgängige Identität und Provenienz von Quelle über Chunk bis Ergebnis,
3. explizite Fähigkeiten und Konfigurations-Fingerprints für Backends und
   Embedding-Provider.

Die Empfehlung ist ausdrücklich **kein universelles RAG-Framework**. Der Core
soll Daten und Verträge definieren; konkrete Query-Planer, Ingestion-Pipelines,
Loaders und LLM-Aufrufe können außerhalb bleiben.

## Priorisierte Findings

### P0: Die Retriever-Signatur ist keine tragfähige Erweiterungsgrenze

Der öffentliche Port akzeptiert nur Text und Limit
([`retrieval/ports.py:27`](retrieval/ports.py#L27)). `parent_id` existiert
dagegen nur in den konkreten SQLite-Methoden
([`retrieval/sqlite/vector.py:97`](retrieval/sqlite/vector.py#L97) und
[`retrieval/sqlite/lexical.py:63`](retrieval/sqlite/lexical.py#L63)). Fusion
und Reranker sehen dieses Argument nicht und können es folglich nicht
weiterreichen.

Das betrifft fast jedes echte RAG-System:

- Tenant, Collection oder Namespace,
- Metadatenfilter und ACLs,
- Zeitbereiche und Dokumenttypen,
- Sprach- oder Quellenfilter,
- ein bereits berechnetes Query-Embedding,
- Deadline, Trace-ID und anwendungsspezifischer Query-Kontext.

Wenn diese Felder später einzeln als Keyword-Argumente ergänzt werden, muss
jeder Retriever und jeder Wrapper gleichzeitig geändert werden. Ein freies
`**kwargs` oder `dict[str, object]` wäre zwar offen, aber untypisiert und führt
zu still ignorierten Optionen.

**Empfehlung:** Vor 1.0 einen unveränderlichen `RetrievalRequest` einführen und
den stabilen Port auf genau ein Request-Objekt ausrichten. Der Core sollte nur
portable Felder fest definieren. Backends dürfen zusätzlich benannte
Capability-Objekte anbieten; Wrapper müssen unbekannte Request-Felder nicht
interpretieren, sondern können das Request-Objekt vollständig weiterreichen.

Ein Filter sollte als kleine, typisierte Ausdrucksstruktur modelliert werden
(`Eq`, `In`, `Range`, `And`, `Or`, `Not`, `Exists`) und nicht als beliebiges
Dictionary. Ein Backend kann nicht unterstützte Ausdrücke dann mit einem
definierten `UnsupportedCapabilityError` ablehnen, statt unbemerkt zu breite
Ergebnisse zu liefern. ACL-Filter dürfen niemals still ignoriert werden.

### P0: Chunking verliert Identität, Position und Quellenbezug

`Chunker.chunk()` nimmt einen nackten String an und liefert nackte Strings
([`retrieval/ports.py:7`](retrieval/ports.py#L7)). Der Quickstart muss IDs und
Dokumente anschließend selbst konstruieren. Dabei gehen Zeichen-/Tokenbereiche,
Seitennummern, Überschriftenpfade und die Beziehung zur Originalquelle
verloren.

Diese Angaben sind keine Komfortfunktion. Sie werden für Zitate, Deep Links,
deduplizierbares Re-Indexing, inkrementelle Updates und Kontextfenster aus
benachbarten Chunks benötigt. Werden sie nur in frei benannten Metadaten
abgelegt, kann kein generischer Retriever oder Reranker darauf vertrauen.

**Empfehlung:** Einen `Chunk`-Datentyp mit mindestens `text`, `ordinal`,
`start_char`, `end_char` und optionalen Tokenpositionen definieren. Zusätzlich
sollte es einen dokumentbewussten Adapter geben, etwa
`chunk_document(document) -> Sequence[DocumentChunk]`, der `source_id`, stabile
Chunk-ID und geerbte Metadaten erhält. Ein einfacher `chunk_text()`-Helper darf
für kleine Anwendungen bestehen bleiben.

Die ID-Strategie muss dokumentiert und austauschbar sein. Eine stabile ID aus
`source_id + chunker_fingerprint + ordinal/content_digest` ist für
inkrementelles Indexieren deutlich belastbarer als eine laufende Nummer allein.

### P0: Embeddings unterscheiden Dokument und Query nicht

Der Provider besitzt nur `embed(texts)`
([`retrieval/ports.py:13`](retrieval/ports.py#L13)). Vector Index und MMR nutzen
daher denselben Aufruf für Dokumente und Suchanfragen. Manche Modelle verlangen
verschiedene Präfixe, Tasks oder Endpunkte für beide Rollen. Diese Information
kann ein Adapter derzeit nicht zuverlässig ableiten.

Außerdem kennt der Port weder Dimensionen noch einen Modell-Fingerprint. Der
SQLite-Index verlangt `dimensions` separat
([`retrieval/sqlite/vector.py:33`](retrieval/sqlite/vector.py#L33)), während der
Cache nur einem manuell vergebenen Namespace vertraut
([`retrieval/sqlite/cache.py:25`](retrieval/sqlite/cache.py#L25)). Ein
Modellwechsel unter demselben Namespace kann dadurch veraltete Vektoren
zurückgeben.

**Empfehlung:** Die Rolle explizit machen, zum Beispiel mit
`embed(texts, *, purpose: EmbeddingPurpose)`, wobei `DOCUMENT` und `QUERY`
mindestens vorgesehen sind. Ein Provider sollte stabile Descriptor-Daten
bereitstellen: `provider`, `model`, `dimensions` und einen Fingerprint aller
vektorrelevanten Einstellungen. Cache und persistenter Index speichern und
validieren diesen Fingerprint selbst.

Batching, Retry und Rate-Limits sollten als Provider-Decorator oder konkrete
Implementierungsdetails möglich bleiben. Sie gehören nicht zwingend in den
kleinsten Port.

### P0: Persistente Collections und Schema-Evolution fehlen

Die SQLite-Tabellennamen sind global fest verdrahtet
([`retrieval/sqlite/vector.py:20`](retrieval/sqlite/vector.py#L20) und
[`retrieval/sqlite/lexical.py:18`](retrieval/sqlite/lexical.py#L18)). Damit kann
eine Datenbank jeweils nur eine Vector- und eine FTS-Konfiguration aufnehmen.
Mehrere Corpora, Embedding-Modelle oder unabhängige Komponenten benötigen
separate Dateien oder eigene Wrapper.

Die Vector-Dimension wird nachträglich geprüft, der FTS-Tokenizer hingegen
nicht. `CREATE VIRTUAL TABLE IF NOT EXISTS` übernimmt eine geänderte
Tokenizer-Konfiguration nicht und meldet den Konflikt auch nicht
([`retrieval/sqlite/lexical.py:165`](retrieval/sqlite/lexical.py#L165)). Für
eine öffentliche Persistenz-API ist stilles Weiterarbeiten mit alter
Konfiguration gefährlich.

**Empfehlung:** Jeder persistente Index erhält eine explizite `collection`.
Eine Manifest-Tabelle speichert Schema-Version, Index-Typ,
Embedding-Fingerprint, Dimensionen, Distanzmetrik, Tokenizer und relevante
Konfiguration. Beim Öffnen gilt eine klare Policy: kompatibel öffnen,
Migration ausführen oder mit `IncompatibleIndexError` abbrechen. Nie still
eine andere Konfiguration benutzen.

Für hybrides Retrieval sollte außerdem entschieden werden, wer Eigentümer des
Dokuments ist. Momentan halten Vector- und FTS-Index getrennte Dokumenttabellen;
ein Fehler zwischen zwei `index()`-Aufrufen kann sie auseinanderlaufen lassen.
Langfristig ist ein gemeinsamer Document Store mit mehreren abgeleiteten
Indexes oder eine explizite, transaktionale Hybrid-Index-Komponente robuster.

### P1: Ergebnisse verlieren Provenienz und Score-Semantik

`RetrievalResult` besteht nur aus Dokument und `float`
([`retrieval/types.py:17`](retrieval/types.py#L17)). Die README weist korrekt
darauf hin, dass Scores verschiedener Retriever nicht vergleichbar sind. Die
Typen verhindern eine falsche Interpretation aber nicht.

RRF verwirft die einzelnen Beiträge. MMR ersetzt den ursprünglichen Score.
Damit fehlen später unter anderem:

- Retriever-/Index-Name und ursprünglicher Rang,
- Score-Art wie Cosine, Distance, BM25, RRF oder Reranker,
- Einzelbeiträge einer Fusion,
- Highlights oder Match-Bereiche,
- Debug-/Erklärdaten und Provider-Rohdaten.

**Empfehlung:** Das Ergebnis als stabiles Envelope modellieren. Ein kleiner
Core könnte `document`, `score`, `score_kind`, `rank`, `source` und
`attributes` enthalten. Fusionsbeiträge sollten als strukturierte Child-Scores
erhalten bleiben. Provider-spezifische Rohdaten gehören in einen klar
benannten, optionalen Bereich und nicht in das Dokument-Metadatenfeld.

Ein portabler `min_score` sollte nur zusammen mit einer definierten
Score-Semantik angeboten werden. Andernfalls bleibt Thresholding Aufgabe des
konkreten Retrievers.

### P1: Fusion löst widersprüchliche Dokumente still nach Reihenfolge auf

RRF dedupliziert ausschließlich über `Document.id`, überschreibt dabei aber
das Dokument bei jedem Treffer
([`retrieval/fusion.py:53`](retrieval/fusion.py#L53)). Liefern zwei Retriever
unter derselben ID unterschiedliche Texte oder Metadaten, gewinnt der später
iterierte Retriever. Das Ergebnis hängt dann von der Konfigurationsreihenfolge
ab, ohne dass der Konflikt sichtbar ist.

**Empfehlung:** Die Identitätsregel explizit machen. Standardmäßig sollte RRF
bei gleicher ID und unterschiedlichem Inhalt einen `DocumentConflictError`
auslösen. Optional kann eine dokumentierte Strategie (`first`, `last`,
`resolver`) injiziert werden. Noch sauberer ist ein gemeinsamer Document Store,
aus dem Fusion nach der ID genau ein kanonisches Dokument lädt.

### P1: `Document` hat backendabhängige Verträge

Das Modell erlaubt `Mapping[str, object]`
([`retrieval/types.py:8`](retrieval/types.py#L8)). Der In-Memory-Index akzeptiert
damit beliebige Python-Objekte, SQLite hingegen nur JSON-serialisierbare Werte
([`retrieval/sqlite/_common.py:83`](retrieval/sqlite/_common.py#L83)). Die
eingefrorene Dataclass friert die Mapping-Inhalte zudem nicht ein.

Auch `created_at` verhält sich unterschiedlich: SQLite erzeugt bei `None` einen
Zeitpunkt, In-Memory lässt `None` stehen. Ein Upsert aktualisiert den Zeitpunkt
nicht ([`retrieval/sqlite/_common.py:37`](retrieval/sqlite/_common.py#L37)). Das
kann gewollt sein, ist aber aktuell kein einheitlicher Vertrag.

**Empfehlung:** Einen rekursiven `JsonValue`-Typ definieren und Metadaten an der
öffentlichen Grenze validieren oder defensiv kopieren. Zeitfelder semantisch
trennen, beispielsweise `created_at` als Quellzeit und `indexed_at` als
Backendzeit. Alle Backends sollten dasselbe Eingabe-/Ausgabeverhalten haben.

Zusätzlich sollte festgelegt werden, ob `Document` eine Quelle oder bereits
die kleinste retriebbare Einheit ist. Für RAG ist ein explizites
`source_id`/`chunk_id`-Modell verständlicher als das überladene `parent_id`.

### P1: ABCs erschweren Adapter unnötig

Alle Ports erben von `ABC`. Ein vorhandener Drittanbieter-Client mit passender
Methode erfüllt den Vertrag deshalb nicht strukturell, sondern muss erben oder
gewrappt werden. Bei einer öffentlichen Integrationsbibliothek sind Wrapper
normal, sollten aber nicht allein wegen Nominal Typing nötig sein.

**Empfehlung:** Schmale öffentliche Ports als `typing.Protocol` definieren.
Konkrete Basisklassen können zusätzlich angeboten werden, wenn gemeinsame
Validierung oder Hilfsmethoden echten Wert liefern. Das hält den Core offen für
LangChain-, LlamaIndex-, Datenbank- und interne Adapter, ohne diese Projekte
als Abhängigkeiten aufzunehmen.

### P1: Lifecycle und Ownership sind nicht komponierbar definiert

SQLite-Komponenten besitzen `close()` und Async-Context-Manager, der
`Retriever`-Port kennt Ressourcen jedoch nicht. Fusion und MMR schließen ihre
Kinder nicht. Der Nutzer muss deshalb die komplette Objektgraph-Struktur
kennen und jede Ressource separat verwalten.

**Empfehlung:** Einheitlich `aclose()` statt `close()` für asynchrone Ressourcen
verwenden und einen optionalen `AsyncCloseable`-Capability-Port definieren.
Kompositionsobjekte sollten standardmäßig keine injizierten Abhängigkeiten
besitzen; ein explizites `owns_children=True` oder ein separater
`RetrievalStack` auf Basis von `AsyncExitStack` kann bequemes Gesamt-Teardown
anbieten. Ownership muss dokumentiert und darf nicht implizit sein.

### P1: Es fehlt eine Conformance-Suite für externe Implementierungen

Die Attraktivität des Pakets hängt davon ab, dass Nutzer eigene Retriever,
Indexes und Provider schreiben. Reine Typen prüfen aber keine semantischen
Zusagen wie Ergebnisreihenfolge, Upsert, Löschen unbekannter IDs,
Limit-Validierung, Embedding-Reihenfolge oder Filterverhalten.

**Empfehlung:** Eine wiederverwendbare `retrieval.testing`-Suite veröffentlichen.
Adapter-Autoren geben Factories an und erhalten parametrisierte Vertragstests
für jeden implementierten Port. Diese Tests sind zugleich die ausführbarste
Definition der stabilen API.

### P2: Bulk-Ingestion, Fehler und Observability brauchen Erweiterungspunkte

`TextIndex.index()` materialisiert eine `Sequence` und liefert immer `None`.
Große Corpora benötigen Batches, Backpressure und eine Möglichkeit, partielle
Fehler oder Statistiken auszuwerten. Gleichzeitig sollte der minimale
Ein-Dokument-/Batch-Fall einfach bleiben.

**Empfehlung:** Den kleinen `index()`-Port behalten, aber optional
`index_many(AsyncIterable[Document]) -> IndexReport` oder eine `IndexSession`
anbieten. `IndexReport` kann Anzahl Upserts, unveränderte Dokumente, Fehler und
Provider-Nutzung tragen. Transaktionsgrenzen müssen pro Backend dokumentiert
sein.

Öffentliche Exception-Typen sollten mindestens Konfiguration, inkompatibles
Schema, nicht unterstützte Capability und temporäre Providerfehler
unterscheidbar machen. Nutzer können dann gezielt migrieren, degradieren oder
wiederholen, statt `ValueError`, `RuntimeError` und Provider-Ausnahmen erraten
zu müssen.

Tracing sollte über Decorators/Middleware oder optionale Hooks möglich sein.
Der Core sollte kein Logging-Framework vorschreiben, aber Request-ID, Dauer,
Kandidatenzahl und Provider-Nutzung beobachtbar machen.

### P2: Öffentliche Oberfläche und Kompatibilitätsziel klarziehen

Die Root-Exports enthalten Ports und Typen, konkrete Implementierungen werden
aus Untermodulen importiert. Das ist sinnvoll, muss aber als verbindliche
Public-API-Policy dokumentiert werden: Nur Namen in `__all__` und dokumentierte
Module erhalten SemVer-Garantien; private Module und Datenbankschema benötigen
eine eigene Kompatibilitätspolitik.

`requires-python = ">=3.13,<3.14"` beschränkt das Paket aktuell auf genau eine
Python-Minor-Version. Für eine öffentliche Bibliothek verkleinert das die
Nutzerbasis stark. Entweder ist dies bewusst und wird erklärt, oder Syntax und
CI-Matrix sollten auf mehrere unterstützte Versionen ausgerichtet werden.

Vor Veröffentlichung sollten außerdem Distribution-Name, Import-Name,
Lizenz, Changelog, Deprecation-Policy und ein `py.typed`-Marker festgelegt
werden. Gerade bei Protocols und Typaliases sind ausgelieferte Typinformationen
Teil des Produkts.

## Empfohlener stabiler Kern

Der folgende Entwurf zeigt die gewünschte Form, nicht zwingend die endgültigen
Namen:

```python
type JsonValue = (
    None | bool | int | float | str |
    list["JsonValue"] | dict[str, "JsonValue"]
)

@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    text: str
    limit: int = 5
    filter: Filter | None = None
    namespace: str | None = None
    context: Mapping[str, JsonValue] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class RetrievalResult:
    document: Document
    score: float
    score_kind: ScoreKind
    source: str | None = None
    rank: int | None = None
    attributes: Mapping[str, JsonValue] = field(default_factory=dict)

class Retriever(Protocol):
    async def retrieve(
        self, request: RetrievalRequest
    ) -> Sequence[RetrievalResult]: ...

class TextIndex(Retriever, Protocol):
    async def index(self, documents: Sequence[Document]) -> IndexReport: ...
    async def delete(self, document_ids: Sequence[str]) -> IndexReport: ...

class EmbeddingProvider(Protocol):
    @property
    def descriptor(self) -> EmbeddingDescriptor: ...

    async def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
    ) -> Sequence[Embedding]: ...
```

Wichtig ist weniger die genaue Anzahl der Felder als die Richtung:

- Request und Resultat sind stabile Envelopes.
- Wrapper reichen Requests verlustfrei weiter.
- Portable Funktionalität ist typisiert.
- Backend-Sonderfunktionen sind explizite Capabilities.
- Keine Option wird still ignoriert.
- Provider- und Indexkonfiguration sind fingerprintbar.

## Was bewusst nicht in den Core gehört

Folgende Funktionen können als optionale Pakete, Beispiele oder externe
Komponenten wachsen, ohne den Kern zu destabilisieren:

- Dateisystem-, Web-, PDF- und Datenbank-Loader,
- vollständige Ingestion-Orchestrierung und Job Queues,
- Prompting, LLM-Aufrufe und Antwortgenerierung,
- konkrete Cross-Encoder und Query-Rewriter,
- Framework-spezifische Adapter,
- UI, Evaluation-Dashboards und Deployment.

Der Core muss diese Anwendungen ermöglichen, aber nicht selbst besitzen.

## Empfohlene Reihenfolge vor 1.0

1. Semantik von `Document`, Chunk-Identität und Metadaten festlegen.
2. `RetrievalRequest`, Filter-AST und Ergebnis-Provenienz einführen.
3. Ports auf Protocols und rollenbewusste Embeddings umstellen.
4. Collection-/Manifest-Modell für persistente Backends implementieren.
5. Lifecycle, Exception-Hierarchie und Capability-Verhalten definieren.
6. Conformance-Suite für externe Adapter veröffentlichen.
7. Erst danach zusätzliche Provider und Backends hinzufügen.

Da das Paket bei `0.1.0` steht, ist jetzt der günstigste Zeitpunkt für diese
Änderungen. Nach einer öffentlichen 1.0 wären insbesondere die
Retriever-Signatur, Dokumentidentität und Embedding-Semantik teuer zu
korrigieren.

## Positive Grundlagen, die erhalten bleiben sollten

- Ein gemeinsamer `Retriever` macht Index, Fusion und Reranker stapelbar.
- Async wird nur dort eingesetzt, wo Provider oder Persistenz I/O benötigen.
- Optionale OpenAI-/SQLite-Abhängigkeiten halten den Core leichtgewichtig.
- Unvergleichbare Scores werden in RRF korrekt über Ränge kombiniert.
- Upsert und idempotentes Löschen sind gute primitive Operationen.
- Die Implementierungen sind klein genug, um als Referenz für eigene Adapter
  zu dienen.

Diese Eigenschaften bilden einen guten stabilen Kern. Die vorgeschlagenen
Envelopes und Capabilities sollen ihn nicht vergrößern, sondern verhindern,
dass notwendige RAG-Funktionen später durch inkompatible Seiteneingänge an ihm
vorbeilaufen.
