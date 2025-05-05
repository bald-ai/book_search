import json
import os

# Mapping from book display name to JSON filename
BOOK_TO_FILE = {
    'Promise of Blood': 'promise_of_blood.json',
    'The Crimson Campaign': 'the_crimson_campaign.json',
    'The Autumn Republic': 'the_autumn_republic.json',
    'Sins of Empire': 'sins_of_empire.json',
    'Wrath of Empire': 'wrath_of_empire.json',
    'Blood of Empire': 'blood_of_empire.json',
}

# Questions, chunk indices, and book associations
QUESTIONS = [
    # (book, question, chunk)
    ('Promise of Blood', "Field Marshal Tamas confronts King Manhouch during the coup in Adopest palace.", 10),
    ('Promise of Blood', "Investigator Adamat searches libraries for information about Kresimir's Promise and finds books with missing pages.", 46),
    ('Promise of Blood', "Field Marshal Tamas's son Taniel and Ka-poel track an escaped Privileged sorceress to Adopest University grounds.", 108),
    ('Promise of Blood', "Field Marshal Tamas is ambushed by soldiers during the Saint Adom's Festival hunt.", 202),
    ('Promise of Blood', "Field Marshal Tamas is ambushed and captured by Kez forces during a hunt at the Saint Adom's Festival.", 219),
    ('The Crimson Campaign', "Adamat interrogates Lord Vetas about his family and Kez connections.", 223),
    ('The Crimson Campaign', "Field Marshal Tamas explains his battle plan at the Fingers of Kresimir to his officers.", 172),
    ('The Crimson Campaign', "Field Marshal Tamas meets with Kez General Beon je Ipille near Hune Dora Forest.", 138),
    ('The Crimson Campaign', "Adamat and Privileged Borbador plan to rescue Taniel Two-Shot and arrest General Ket.", 357),
    ('The Crimson Campaign', "Adamat rescues Faye from Lord Vetas's headquarters.", 198),
    ('The Autumn Republic', "Field Marshal Tamas uncovers General Hilanska's treason and connection to the Kez.", 62),
    ('The Autumn Republic', "Inspector Adamat investigates the bombing at the Holy Warriors of Labor headquarters in Adopest.", 152),
    ('The Autumn Republic', "Taniel Two-shot searches for Ka-poel in the mountains of southwest Adro.", 31),
    ('The Autumn Republic', "Privileged Bo is severely injured during an attack by Kez Privileged.", 133),
    ('The Autumn Republic', "Field Marshal Tamas directs artillery fire at a weak point in Budwiel's wall.", 253),
    ('Sins of Empire', "Privileged Robson investigates a mysterious Dynize obelisk near Landfall where workers went mad.", 3),
    ('Sins of Empire', "Fidelis Jes, Grand Master of the Blackhats, assigns Michel Bravis to investigate fifteen missing Iron Roses and the Sins of Empire pamphlet.", 28),
    ('Sins of Empire', "Gregious Tampo secures Benjamin Styke's release from the labor camp and hires him to join General Vlora Flint's mercenary company.", 35),
    ('Sins of Empire', "Benjamin Styke meets General Vlora Flint and Colonel Olem at a café in Landfall and is hired to investigate Palo dragonmen.", 55),
    ('Sins of Empire', "Michel Bravis shows General Vlora Flint the difficult conditions and layout of Greenfire Depths.", 64),
    ('Wrath of Empire', "Who does Ka-Sedial order to hunt down Ben Styke?", 4),
    ('Wrath of Empire', "What happens when General Vlora Flint's forces battle the Dynize army commanded by Bar-Levial?", 24),
    ('Wrath of Empire', "What does Michel Bravis uncover and what action does he take in the Dynize Household?", 183),
    ('Wrath of Empire', "Who attempts to destroy a godstone in Yellow Creek?", 303),
    ('Wrath of Empire', "Who does Colonel Ben Styke kill during a cavalry charge?", 198),
    ('Blood of Empire', "Who does Ben Styke kill and where does he leave the head?", 292),
    ('Blood of Empire', "What does Vlora Flint discover near the Yellow Creek godstone site?", 111),
    ('Blood of Empire', "Who does Michel and Ichtracia collaborate with to create propaganda against Sedial?", 44),
    ('Blood of Empire', "How does Vlora Flint transport her army towards Ka-Sedial's fortress?", 342),
    ('Blood of Empire', "What does Ben Styke use to disperse a mob in the Dynize capital?", 334),
]

# Preload all chunk data for efficiency
chunk_data = {}
for book, filename in BOOK_TO_FILE.items():
    path = os.path.join('chunks_json', filename)
    with open(path, 'r', encoding='utf-8') as f:
        chunk_data[book] = json.load(f)
    # Build a dict for fast lookup
    chunk_data[book] = {c['chunk_index']: c['text'] for c in chunk_data[book]}

# Group questions by book
from collections import defaultdict
book_questions = defaultdict(list)
for idx, (book, question, chunk_idx) in enumerate(QUESTIONS, 1):
    book_questions[book].append((idx, question, chunk_idx))

# Prepare output
output_lines = []
for book in BOOK_TO_FILE:
    output_lines.append(f"\n{book}")
    for idx, question, chunk_idx in book_questions[book]:
        chunk_text = chunk_data[book].get(chunk_idx, '[Chunk not found]')
        output_lines.append(f"  {idx}. {question}\n    Chunk {chunk_idx}:\n\n    Chunk Text:\n{chunk_text}\n")

# Write to file
with open('Petra_questions.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines)) 