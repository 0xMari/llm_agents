# TL;DR Progetto Agent Lab Viaggi

Cartella progetto:

```bash
cd /Users/marianionutcioancaradu/maria/test
source .venv/bin/activate
```

Ambiente verificato:

```text
Python 3.12.8
Pydantic 2.13.4
```

Comandi utili:

```bash
python main.py
python -m unittest test_workflow.py
```

Stato test:

```text
Ran 4 tests
OK
```

## File Attuali

`models.py` contiene i contratti Pydantic:

- `TripRequest`
- `SuggestedTimePeriod`
- `FlightOffer`
- `HotelOffer`
- `TripProposal`
- `TripSearchResult`
- `PlanningError`
- `EventType`
- `ReasonCode`
- `DecisionEvent`
- `ProposalComment`
- `DecisionTrace`

Hai gia aggiunto reason code e commenti strutturati, pensati per essere compatibili con un futuro LLM.

`data.py` contiene dati finti:

- 4 voli
- 2 hotel

`agents.py` contiene agenti simulati:

- `suggest_periods()`
- `find_flights()`
- `find_hotels()`
- `list_hotels()`
- `suggest_activities()`

`main.py` e il coordinatore:

- analizza i periodi
- genera combinazioni volo + hotel
- esclude combinazioni fuori budget con `ReasonCode.OVER_BUDGET`
- aggiunge commenti alle proposte valide
- ordina per score del periodo, rating hotel e prezzo totale piu basso
- restituisce massimo 3 proposte

`test_workflow.py` verifica:

- le proposte rispettano il budget
- le proposte hanno commenti con `WITHIN_BUDGET`
- `min_days > max_days` genera `ValidationError`
- budget basso restituisce `PlanningError`
- il risultato contiene massimo 3 proposte

## Prossimo Passo Consigliato

Piccola pulizia:

- rimuovere `find_hotels` dall'import di `main.py`, perche ora non viene usato
- sistemare gli spazi tipo `event_type= EventType...` in `event_type=EventType...`
- aggiungere un nuovo modello `TravelExplanation`, che sara il primo punto di aggancio per un futuro LLM

## Subito Dopo: introdurre LLM senza rompere il sistema

Il primo inserimento di un LLM non dovrebbe decidere:

- quali offerte sono valide
- quali proposte sforano il budget
- quali record vanno esclusi

Quelle parti devono restare deterministiche dentro Python.

Il primo uso sensato del modello e invece:

- spiegare una proposta gia valida
- trasformare `ReasonCode` e commenti strutturati in testo leggibile
- evidenziare tradeoff come prezzo vs comfort

Questo approccio e il meno rischioso, perche:

- il modello lavora su dati gia filtrati
- puoi verificare facilmente se l'output inventa qualcosa
- se fallisce, puoi sempre tornare a una spiegazione deterministica

## Architettura Consigliata

Mantieni separati tre livelli:

- agenti deterministici, che cercano e combinano dati
- orchestratore, che applica regole hard e costruisce la trace
- agente LLM, che genera spiegazioni o sintesi

Flusso consigliato:

1. `main.py` produce una `TripProposal` valida e una `DecisionTrace`
2. un adapter compatto prepara solo i fatti essenziali per il modello
3. il modello restituisce un `TravelExplanation`
4. il risultato finale unisce dati strutturati e testo umano

Regola pratica:

- l'LLM non deve fare aritmetica critica
- l'LLM non deve essere la fonte di verita del budget
- l'LLM non deve sostituire i filtri di business

## Il Primo Agente LLM Da Aggiungere

Non partire da un generico "planner agent".

Il primo agente dovrebbe essere qualcosa di molto stretto, ad esempio:

- `proposal_explainer`

Responsabilita:

- leggere una proposta gia accettata
- spiegare perche e stata tenuta
- descrivere il compromesso principale
- generare un testo breve, coerente e verificabile

Da evitare nel primo step:

- tool calling
- ricerca libera di voli e hotel
- memoria conversazionale persistente
- agenti che si passano istruzioni vaghe tra loro

## Contratto Da Definire Prima Del Prompt

Prima ancora di scrivere il prompt, definisci bene input e output.

Input minimo utile:

- destinazione
- intervallo date scelto
- prezzo volo
- prezzo hotel
- totale
- budget utente
- score del periodo
- rating hotel
- `ReasonCode` principali
- commenti strutturati gia presenti

Output atteso del futuro `TravelExplanation`:

- un riassunto breve
- la ragione principale della proposta
- il tradeoff piu importante
- un warning opzionale, se serve

Vincoli forti:

- output strutturato, non testo libero puro
- nessun dato non presente nell'input
- frasi brevi
- fallback se la risposta non e parseabile

## Come Arrivare All'LLM In Modo Pulito

Sequenza consigliata:

1. aggiungi `TravelExplanation` come nuovo modello Pydantic
2. crea una versione finta e deterministica dell'explainer
3. collega l'orchestratore a questa interfaccia, non direttamente a un provider LLM
4. solo dopo sostituisci l'implementazione finta con una reale

Il vantaggio e importante:

- la pipeline si stabilizza prima del modello
- puoi testare il contratto senza costo LLM
- il cambio provider in futuro diventa semplice

## Test Da Preparare Quando Entrera LLM

Quando arriverai davvero al modello, i test piu utili saranno:

- output valido e parseabile
- nessuna contraddizione con il budget calcolato dal codice
- nessuna invenzione di hotel, prezzi o date
- fallback corretto se il modello risponde male
- stabilita minima con temperatura bassa

Non serve testare se il modello "ha gusto".
Serve testare se resta dentro i confini del contratto.

## Quando Ha Senso Passare A Piu Agenti Veri

Finche il sistema fa:

- ricerca dati
- applicazione regole
- ranking base
- spiegazione finale

...un orchestratore piu un solo agente LLM bastano.

Ha senso passare a piu agenti solo quando hai ruoli davvero distinti, per esempio:

- un agente che spiega la proposta
- un agente che confronta due proposte valide
- un agente che converte trace tecniche in testo per l'utente

Se introduci piu agenti troppo presto, aumentano:

- complessita
- punti di errore
- costo di debugging
- ambiguita tra responsabilita

## Direzione Consigliata

La soglia giusta per dire "ora introduco LLM" non e quando vuoi piu automazione.
E quando hai gia una pipeline stabile e vuoi migliorare:

- leggibilita
- motivazione delle scelte
- qualita della spiegazione finale

In breve: prima contratti chiari, poi adapter, poi un explainer LLM piccolo e controllato.
