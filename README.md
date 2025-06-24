# Mitől lesz jó egy képleírás?

Ebben a repóban olyan kódok és kimeneteik vannak, melyekkel azt vizsgáltuk, hogy képekhez egyedileg generált leírsok mennyire illeszkednek, ezek hogyan alakítják a látens tér struktúráját.

---

## Mappastruktúra


---

## data/

Itt található minden adat:

- `captions_till_50.json`: Az első 50 osztály minden képéhez caption
- `custom_captions.json`: Minden osztályhoz caption
- `short_custom_captions.json`: Minden osztályhoz rövid caption
- `paraphrased_classes_till_50.json`: Átfogalmazott leírások a `custom_captions.json` alapján
- `image_embeddings_siglip.pt`: Embeddingek, amiket a SigLIP számolt
- `image_info_siglip.pt`: A képekhez tartozó metaadatok

---

## notebooks/

- `create_image_embedding.py`: Ezzel lehet modellt választani, aztán a többi kódot futtatni rajta
- `online_presentation.ipynb`: Leginkább vizualizációk vannak benne
- `tablazatok.ipynb`: Itt készül minden táblázat

---

## tables/

Mérési eredmények különböző metrikák mentén:

- `class_average_precision.csv`: Osztályszintű átlagos precíziós értékek
- `class_precision_at_class_size.csv`: Precízió osztályméret szerint
- `class_recall_cutoffs.csv`: Osztályszintű visszahívási küszöbök
- `class_top1_similarity.csv`: Legmagasabb osztály-szintű hasonlóság
- `image_average_precision.csv`: Kép-szintű átlagos precízió
- `image_recall_at_k.csv`: Kép-szintű Recall@k értékek
- `image_top1_similarity.csv`: Legjobb kép-kép pár hasonlósága képenként
- `inter_class_similarities.json`: Különböző osztályok közötti hasonlóságok
- `intra_class_similarities.json`: Egyes osztályokon belüli hasonlóságok

---
