# rltraffic × Claude Code — setup i system pracy

Dwie części: **A. wdrożenie** (raz, ~20 min) i **B. system pracy** (codziennie).
Część B jest ważniejsza. Setup bez dyscypliny pracy to tylko szybszy sposób na produkowanie kodu,
którego nikt nie przeczytał.

---

# CZĘŚĆ A — WDROŻENIE

## A1. Instalacja (w WSL, nie w PowerShellu)

```bash
node --version                          # brak? -> Node LTS (nvm albo apt)
npm install -g @anthropic-ai/claude-code
cd ~/rltraffic && claude                # pierwsze uruchomienie: logowanie w przeglądarce
```
W sesji: `/doctor` (diagnostyka), `/status` (model, katalog, uprawnienia).
Dokumentacja: https://docs.claude.com/en/docs/claude-code/overview

Repo zostaje w Linux FS (`~/rltraffic`), nigdy `/mnt/c/...` — masz to w Decisions Log, a przy
kilkuset operacjach na plikach w sesji różnica jest bolesna.

## A2. Przeniesienie paczki z Windows do WSL

Pobrany z czatu plik ląduje w Downloads **po stronie Windows**. WSL widzi go pod `/mnt/c/`.
Nie musisz go nigdzie kopiować — rozpakuj wprost do repo.

```bash
# 1. znajdz swoj katalog uzytkownika Windows (moze sie roznic od loginu WSL)
ls /mnt/c/Users/

# 2. podstaw wlasciwa nazwe i sprawdz, ze pliki sa na miejscu
WINUSER=filip
ls -la /mnt/c/Users/$WINUSER/Downloads/rltraffic_claude_setup_v2.tar.gz
ls -la /mnt/c/Users/$WINUSER/Downloads/project_master_plan_rltraffic*

# 3. zobacz, co jest w srodku, ZANIM rozpakujesz do repo
tar -tzf /mnt/c/Users/$WINUSER/Downloads/rltraffic_claude_setup_v2.tar.gz
#    wszystko musi byc pod jednym katalogiem rltraffic_claude_setup/ -
#    dlatego dziala --strip-components=1 w nastepnym kroku

# 4. rozpakowanie do repo
cd ~/rltraffic
git status                       # drzewo ma byc czyste, jestes na main
tar -xzf /mnt/c/Users/$WINUSER/Downloads/rltraffic_claude_setup_v2.tar.gz --strip-components=1
chmod +x scripts/claude_guard.sh
mkdir -p docs/plans

# 5. master plan do repo - jedno zrodlo prawdy zamiast pliku w Downloads.
#    Przegladarka mogla dopisac (1)/(2)/(3) do nazwy, wiec skopiuj JAWNIE ten wlasciwy,
#    nie globem - glob dopasowujacy kilka plikow nadpisze cel po cichu.
cp "/mnt/c/Users/$WINUSER/Downloads/<dokladna-nazwa-planu>.md" docs/PROJECT_PLAN.md
head -3 docs/PROJECT_PLAN.md     # sprawdz, ze to naprawde plan, a nie stara wersja

# 6. gitignore + commit
cat gitignore_additions.txt >> .gitignore && rm gitignore_additions.txt
git add -A && git commit -m "chore: Claude Code setup, contracts v1.1, brief P1 v2"
```

Rozpakowanie nadpisze `CLAUDE.md`, `.claude/`, `docs/CONTRACTS.md`, `docs/briefs/`, `docs/returns/`
i `scripts/claude_guard.sh`. Reszta repo pozostaje nietknieta - tar dokleja, nie czysci katalogow.
Jesli rozpakowywales wczesniejsza wersje paczki, usun najpierw
`docs/briefs/ADDENDUM_A_PATCH.md` i `docs/briefs/BRIEF_01_DELTA.md` - v2 ich nie zawiera, wiec
same nie znikna, a sa sprzeczne z Briefem #1 v2.

Paczka jest już scalona ze zwrotką master chatu — `settings.json` i `claude_guard.sh` w wersji
dwutrybowej, `docs/CONTRACTS.md` z C6 v1.1, `BRIEF_01_v2` w `docs/briefs/`, a nieaktualne
`ADDENDUM_A_PATCH.md` i `BRIEF_01_DELTA.md` usunięte. Nie musisz nic składać ręcznie.

**Jedna zmiana względem tego, co przysłał master chat:** w `settings.json` ścieżki hooków to teraz
`bash "$CLAUDE_PROJECT_DIR/scripts/claude_guard.sh"` zamiast ścieżki względnej. Hooki nie zawsze
odpalają się z rootu repo — przy ścieżce względnej hook po cichu nie znajdzie skryptu, a wtedy nie
masz żadnego zabezpieczenia i nie dowiesz się o tym.

## A3. Weryfikacja — obowiązkowa, 3 minuty

Nie „sprawdziłem że skrypt działa", tylko „sprawdziłem że hook się odpala".

```bash
# 1. skrypt sam w sobie
echo "# test" >> envs/base_traffic_env.py
bash scripts/claude_guard.sh --frozen-only ; echo "exit=$?"   # oczekiwane: exit=2 + BLOCKED
git checkout -- envs/base_traffic_env.py
```

```bash
# 2. hook w sesji — jedyny prawdziwy test
claude
```
```
/hooks                    # oba wpisy PostToolUse widoczne?
/permissions              # deny na envs/**, agent/base.py, ...?
/agents                   # contract-reviewer, repo-cartographer, citation-verifier?
```
Potem, w sesji, poproś: *„dopisz komentarz `# probe` na końcu `envs/base_traffic_env.py`"*.
Poprawny wynik: uprawnienia blokują edycję **albo** hook zwraca BLOCKED. Jeśli plik zmienił się bez
protestu — masz dziurę, popraw zanim ruszysz dalej.

## A4. Terminal czy VS Code

- **VS Code (WSL Remote) + rozszerzenie Claude Code** — praca codzienna. Rozszerzenie instalujesz po
  stronie WSL. Repo otwierasz przez `code .` z `~/rltraffic`. Diffy inline są tu istotne: to jest kod,
  którego wyniki trafią do tabel w artykule.
- **Osobny terminal + `tmux`** — wszystko powyżej kilku minut: P2.1 (MAPPO ≥500 epizodów),
  P2.2 (kampania korpusu). Nigdy w sesji Claude Code. Agent czyta logi, nie trzyma procesu.

---

# CZĘŚĆ B — SYSTEM PRACY

## B1. Warstwy — co gdzie mieszka

Twój łańcuch *Goals → Requirements → Spec → Implementation* jest już zaimplementowany, tylko rozbity
na dwa narzędzia. Claude Code nie musi Cię przepytywać o cele, bo cele są w repo.

| Warstwa | Artefakt | Gdzie powstaje |
|---|---|---|
| Goals | claims C1–C3, headline contribution | Master chat (claude.ai) |
| Requirements | plan §5–§7, kontrakty §4 | Master chat → `docs/PROJECT_PLAN.md`, `docs/CONTRACTS.md` |
| Spec | brief zadania | Master chat → `docs/briefs/BRIEF_XX.md` |
| Implementation | kod + testy | **Claude Code** |
| QA | recenzja kontraktowa | **Claude Code** (`/review`, subagent w świeżym kontekście) |
| Release | merge + Return Packet | Claude Code → `/handoff` → Master chat |
| Iterate | aktualizacja planu, następny brief | Master chat |

**Jeden brief = jeden sprint = jedna sesja = jeden branch = jeden Return Packet.** To jest jednostka
pracy. Brief §7 limituje ją do ≤2 plików źródłowych + testy — to nie formalność, tylko granica
poniżej której recenzja jest w ogóle wykonalna.

## B2. Pętla jednego zadania

```
Shift+Tab -> plan mode          (zawsze; widać w stopce)
/task P1
  Faza 1 EXPLORE   -> BRAMKA 1: czytasz rozbieżności brief↔repo, akceptujesz
  Faza 2 PLAN      -> BRAMKA 2: czytasz docs/plans/P1.md, akceptujesz
  Faza 3 CODE      -> BRAMKA 3: czytasz testy ZANIM powstanie implementacja
  Faza 4 COMMIT    -> Return Packet w docs/returns/P1.md
/review P1                       -> contract-reviewer, świeży kontekst, read-only
  (poprawki -> wracasz do fazy 3)
merge do main                    -> dopiero po PASS
/handoff P1                      -> paczka dla Master chatu + co dziedziczy następna faza
/clear
```

### Dlaczego bramka 3 jest najważniejsza
Testy w Briefie #1 kodują konwencję alignmentu (`obserwacje T+1, decyzje T, r_t z wiersza t+1`).
Jeśli implementacja powstanie pierwsza, testy zostaną napisane tak, żeby ją potwierdzić — i off-by-one
wejdzie do formatu danych, gdzie kosztuje regenerację korpusu. Test 3 z briefu (dokładna równość
`-np.sum(lane_waiting_vehicle_count[t+1])` vs `global_reward[t]`) jest jedynym miejscem, w którym ta
konwencja jest naprawdę egzekwowana. Przeczytaj go osobiście, zanim powstanie logger.

### Reguła 95%
Jest w `CLAUDE.md`: agent ma wypisać założenia z poziomem pewności i **pytać zamiast zgadywać**
poniżej ~95%. Nie kasuj tego jako gadulstwa — w tym projekcie złe założenie nie wywala się z błędem,
tylko produkuje prawdopodobną liczbę.

## B3. Kontekst — jedna zasada i cztery nawyki

> **Repo jest pamięcią. Okno kontekstu jest brudnopisem.**
> Cokolwiek musi przetrwać, ląduje na dysku: plan w `docs/plans/`, konwencja w docstringu, wynik
> w `docs/returns/`. Nic ważnego nie zostaje wyłącznie w rozmowie.

1. **`/clear` między zadaniami. Zawsze.** Kontekst z P1 w sesji P2 to nie oszczędność, to źródło
   cichych założeń, których nikt nie wypowiedział.
2. **Czytanie deleguj do subagentów.** `/explore` i `repo-cartographer` czytają pliki we własnym
   kontekście i zwracają wniosek z `path:line`. Główna sesja dostaje odpowiedź, nie źródła. To jest
   najskuteczniejsza technika oszczędzania kontekstu, jaką masz.
3. **`/compact` tylko w środku zadania i zawsze z instrukcją**, np.:
   `/compact zachowaj: konwencję alignmentu, podjęte decyzje, ostatni output pytest; wyrzuć: treść plików, które są już na dysku`.
   Bezinstrukcyjny compact wyrzuca dokładnie to, co było trudne do ustalenia.
4. **Jeśli po raz drugi tłumaczysz agentowi to samo — to nie należy do rozmowy, tylko do `CLAUDE.md`.**

## B4. Antywzorce (każdy realnie kosztowałby ten projekt)

| Nie rób | Dlaczego |
|---|---|
| „testy powinny przechodzić" bez uruchomienia | `CLAUDE.md` §2 tego zakazuje; egzekwuj w Return Packecie |
| poprawianie testu, żeby przeszedł | domyślna hipoteza: to kod jest zły. Jeśli test jest zły — stop i decyzja Twoja |
| pominięcie `/review`, „bo mała zmiana" | ten sam kontekst, który pisał kod, jest najgorszym recenzentem tego kodu |
| dwa zadania w jednej sesji | patrz B3.1 |
| długa symulacja w sesji agenta | zjada kontekst i blokuje sesję; `tmux` |
| always-on skille typu ponytail/caveman | YAGNI skasuje state machine, hash epizodu i guard na zmianę pasów — czyli dokładnie to, co broni korpusu |

## B5. Praca równoległa (przyda się przy P2)

Gdy P2.0 (randomizer) i P2.1 (trening MAPPO) się nałożą, nie prowadź dwóch sesji w jednym katalogu:

```bash
git worktree add ../rltraffic-p20 -b task/p20-randomizer
cd ../rltraffic-p20 && cp -r ~/rltraffic/.claude .   # hooki nie wędrują do worktree same
```
Każdy worktree = własna sesja, własny branch, zero kolizji.

## B6. Checklista dnia

- [ ] `git status` czysty, jestem na `main`
- [ ] wiem, który brief robię i że jest **jedyną** wersją prawdy dla tego zadania
- [ ] plan mode włączony
- [ ] po zadaniu: `/review` → PASS → merge → `/handoff` → `/clear`
- [ ] Return Packet wklejony do Master chatu

---

## Dodatek: dwie znane słabości guarda (do decyzji, nie zmieniałem)

`claude_guard.sh` jest w wersji master chatu i został przez niego przetestowany — zostawiłem
bez zmian. Dwie rzeczy warto rozważyć, gdy zaczną boleć:

1. **`--tests-only` odpala `pytest tests`** — całe drzewo testów. Jeśli w `tests/` są testy wymagające
   SUMO/CityFlow, hook będzie zwracał exit 2 z powodów niezwiązanych ze zmianą i agent zacznie
   „naprawiać" cudze testy. Fix: zmienna `CLAUDE_GUARD_TESTS` z domyślną wartością równą plikowi
   testowemu bieżącego zadania.
2. **Parsowanie `git status --porcelain` przez `awk '{$1=""}'`** gubi zmiany nazw (`R  old -> new`) i
   pliki ze spacjami. Zamrożony plik przemianowany zamiast edytowanego przejdzie niezauważony.
   Fix: `cut -c4-` albo `git diff --name-only HEAD`.

Oba są mało prawdopodobne dziś i tanie do naprawy później — dlatego są tutaj, a nie w kodzie.
