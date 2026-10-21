# Fixture SerpAPI reali

`autocomplete_rome.json` deriva dalla risposta reale fornita dall'utente nella
conversazione. Conserva solo `name`, `type`, `id` e gli ID in `airports`, senza
aggiungere i `type` assenti. E una fixture ridotta per riprodurre il problema
dei suggerimenti non-citta, non una nuova chiamata live.
La fixture hotel deve ancora essere registrata: non vengono inventate risposte
per simulare una registrazione reale.

## Registrazione manuale

Dalla radice del progetto, configura `SERPAPI_API_KEY` nel terminale e avvia:

```bash
.venv/bin/python record_serpapi.py --origin "Rome" --origin-city-id "/m/06c62" --destination "Lisbon"
```

Il comando usa Roma e Lisbona, un adulto e date future. Riutilizza la prova
SerpAPI esistente con una sola andata da approfondire e hotel abilitati. Sono
consentite al massimo cinque chiamate: due autocomplete, andate, ritorni e hotel.
Non chiama OpenRouter. `--help` non effettua ricerche.

L'ID di Roma e verificato nella risposta fornita dall'utente. In caso di omonimie
usa `--origin-city-id` o `--destination-city-id` con uno degli ID mostrati
dall'autocomplete: il programma ne verifica la presenza tra le citta restituite,
senza scegliere il primo risultato o accettare ID arbitrari.

Il file predefinito e `fixtures/serpapi/hotels_real.json`. Per un'altra prova:

```bash
.venv/bin/python record_serpapi.py --origin "Rome" --origin-city-id "/m/06c62" --destination "Lisbon" --adults 2 --output fixtures/serpapi/hotels_two_adults.json
```

Un file esistente non viene sovrascritto e viene rilevato prima delle chiamate.
Non servono entrambe le prove per iniziare: ogni esecuzione live puo consumare quota.

## Contenuto e limiti

La fixture contiene provenienza/data di registrazione, query e risposta ridotta
prima della conversione in `HotelOffer`. Conserva nomi pubblici degli hotel,
rating e prezzi numerici originali, compresi i decimali. Esclude URL, token,
chiavi API, metadati della ricerca e altri campi non utilizzati da questo test.
Anche gli oggetti dei prezzi sono filtrati con una allowlist.

Controlla il contenuto prima di includerlo in Git: destinazione, date e ospiti
restano nella query. Usa itinerari di prova, non richieste personali sensibili.
Il registratore supporta soltanto risposte elenco `properties` non vuote in EUR.
Errori, risultati vuoti e risposte property-details non diventano fixture positive.
Una risposta registrata puo comunque far fallire il parser: analizza la differenza,
non correggere artificialmente il JSON per far passare il test.

## Replay offline

```bash
.venv/bin/python -B -m unittest test_hotels.RecordedHotelFixtureTest -v
.venv/bin/python -B -m unittest discover -q
```

Il test legge tutti i file `hotels_*.json` in questa cartella, usa il provider e
il parser reali e sostituisce solo la risposta del client HTTP. La rete e bloccata.
Senza fixture il test e esplicitamente skipped; non registra nulla automaticamente.
I prezzi storici non devono essere aggiornati: il test verifica la conversione,
non disponibilita e tariffe correnti. Autocomplete, voli, redirect e il flusso
completo con LLM non sono registrati da questo primo strumento.
La fixture autocomplete fornita manualmente viene invece verificata dai test
in `test_flights.py`, anche per preservare la richiesta di chiarimento tra Roma
e Rome in Georgia.
