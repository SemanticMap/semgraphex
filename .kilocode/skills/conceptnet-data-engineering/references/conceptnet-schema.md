# ConceptNet 5 Schema and API Reference

## Authoritative sources

- Repository: https://github.com/commonsense/conceptnet5
- Wiki: https://github.com/commonsense/conceptnet5/wiki
- Downloads: https://github.com/commonsense/conceptnet5/wiki/Downloads
- Edges: https://github.com/commonsense/conceptnet5/wiki/Edges
- Relations: https://github.com/commonsense/conceptnet5/wiki/Relations
- URI hierarchy: https://github.com/commonsense/conceptnet5/wiki/URI-hierarchy
- API: https://github.com/commonsense/conceptnet5/wiki/API
- Languages: https://github.com/commonsense/conceptnet5/wiki/Languages
- License guidance: https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet

The official repository exceeds the project requirement of five stars or forks by a wide margin: search results at skill creation time showed about 3,000 stars and more than 350 forks.

## ConceptNet 5.7 assertion dump

The official 5.7.0 assertion artifact is:

`https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz`

Despite the `.csv.gz` suffix, decompressed records are tab-separated. Each row has five fields:

1. assertion URI;
2. relation URI;
3. start node URI;
4. end node URI;
5. JSON metadata.

Representative shape:

```text
/a/[…]	/r/Antonym	/c/ab/…	/c/ab/…	{"dataset":"/d/wiktionary/en","license":"cc:by-sa/4.0","sources":[…],"weight":1.0}
```

The dump is the preferred input for complete or reproducible experiments. Stream it with gzip-aware I/O and parse the fifth field with a JSON parser.

## Edge fields

The API represents assertions with named fields:

| Field | Meaning | Engineering rule |
|---|---|---|
| `@id` / `uri` | Unique assertion URI | Preserve for lineage; do not parse when direct fields exist. |
| `rel` | Relation object or URI | Normalize to canonical `/r/...` URI. |
| `start` | First argument | Preserve direction. |
| `end` | Second argument | Preserve direction. |
| `weight` | Positive assertion strength | Preserve raw; do not interpret as probability. |
| `sources` | Factors contributing the assertion | Retain for provenance and source ablations. |
| `license` | Creative Commons URI | Retain and audit redistribution obligations. |
| `dataset` | Build-time dataset/batch URI | Retain for confound and provenance analysis. |
| `surfaceText` | Original text, sometimes null | Optional evidence/presentation field; do not require it. |

API `start`, `end`, and `rel` values are objects containing `@id`; dump columns are already URIs. Normalize both representations into one internal schema.

## URI hierarchy

Important namespaces include:

- `/c/{language}/{term}`: natural-language concept;
- `/c/{language}/{term}/{pos}`: part-of-speech-specific concept;
- more specific concept URIs may add a sense disambiguation segment;
- `/r/{Relation}`: relation;
- `/a/[...]`: assertion;
- `/d/...`: dataset;
- `/s/...`: source, contributor, process, resource, or activity;
- `/and/[...]`: conjunction of source assertions.

Concept hierarchy is prefix-like. Looking up `/c/it/esempio` can include more specific senses such as `/c/it/esempio/n`, but the URIs remain distinct identifiers. Do not collapse them by default.

Language codes follow BCP 47-style two- or three-letter identifiers. Parse the second segment after `/c/`; do not assume all useful languages have two-letter codes.

Phrase URIs generally replace spaces with underscores, but prefer the API `/uri?language=...&text=...` endpoint when canonicalizing arbitrary user input. For dump processing, trust the canonical URIs already present.

## Relation semantics

The official 5.7 relation page lists 34 relations. High-priority project relations include:

| Relation | Direction/meaning notes |
|---|---|
| `/r/RelatedTo` | General positive relationship; symmetric; broad and often dominant. |
| `/r/FormOf` | Inflected form points to root form; directional. |
| `/r/IsA` | Subtype or instance points to broader type; directional. |
| `/r/PartOf` | Part points to whole; directional. |
| `/r/HasA` | Possessor/whole points to possessed part; often reverse-like to `PartOf`, but not identical. |
| `/r/UsedFor` | Object/activity points to its purpose; directional. |
| `/r/CapableOf` | Entity points to a typical action; directional. |
| `/r/AtLocation` | Concept points to a typical/inherent location; directional. |
| `/r/Causes` | Typical cause points to effect; directional. |
| `/r/HasProperty` | Entity points to property; directional. |
| `/r/Synonym` | Similar meanings or translations; symmetric. |
| `/r/Antonym` | Relevant opposites; symmetric. |
| `/r/DistinctFrom` | Distinct alternatives; symmetric. |
| `/r/LocatedNear` | Typical proximity; symmetric. |
| `/r/ExternalURL` | Concept links to an external Linked Data URL; not a concept-to-concept semantic edge. |

Consult the official relation table before implementing a complete symmetry registry. Keep relation meaning available after any projection.

## API access patterns

Base URL: `http://api.conceptnet.io` in the wiki examples. Prefer HTTPS where supported by the runtime.

### Lookup

Append a known URI:

```text
/c/en/example
/r/Antonym
/s/contributor/omcs/dev
/a/[…]
```

Node responses contain `@context`, `@id`, `edges`, and a `view` pagination object. Follow `nextPage` until absent when complete retrieval is required.

### Query

Use `/query` with these filters:

- `start`;
- `end`;
- `rel`;
- `node` for either endpoint;
- `other` for the opposite endpoint from `node`;
- `sources`.

Include `limit` intentionally and still inspect pagination.

### URI conversion

Use `/uri?language=en&text=french+toast` to map natural-language input to a concept URI.

### Related and relatedness

`/related/{concept-uri}` and `/relatedness?node1=...&node2=...` use reduced ConceptNet Numberbatch embeddings. Their scores are embedding relatedness, not assertion weights, edge existence, or graph transition probabilities.

### Rate limits

The wiki documents 3600 requests per hour and bursts of 120 requests per minute. Calls to `/related` and `/relatedness` count as two requests. Design average traffic below one request per second and cache responses where permitted.

## Weight semantics

The official edge documentation calls weight the strength with which an edge expresses an assertion; the API documentation describes it as believability that tends to rise with more or more reliable sources. Typical weight is 1.0, all documented weights are positive, and values can be above or below 1.

This does not establish calibration as probability, frequency, conditional probability, causal effect, or Markov transition. Preserve the raw value and make analytical transforms explicit.

## Licensing

The complete ConceptNet data is available under CC BY-SA 4.0 according to the copying/sharing wiki. Attribution and share-alike both matter. Individual API edges may expose licenses such as `cc:by/4.0` or `cc:by-sa/4.0`, but reuse of ConceptNet as a whole must follow the complete-data guidance.

For research, the wiki requests citation of:

Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. “ConceptNet 5.5: An Open Multilingual Graph of General Knowledge.” AAAI 31.

Perform a license review before redistributing filtered, transformed, or merged datasets, and provide user-visible attribution.
