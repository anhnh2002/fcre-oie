from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

extract_trigger_prompt = """
Given a piece of text, two entities subject, object (not ordered) and corresponding relation type between two entities, extract the relation trigger in the form of [Subject, Relation trigger, Object] from it.
Here are some examples:
Example 1:
Text: "he passed away on saturday ."
Subject, Object (not ordered): "he", "saturday"
Relation type: "person date of death"
Complete triplets: ["he", "passed away on", "saturday"]

Example 2:
Text: "as a substantial shareholder in cnac 's subsidiary air china , cathay pacific said late monday it would give serious consideration to joining cnac and form a strategic partnership with china eastern ."
Subject, Object (not ordered): "cnac", "cathay pacific"
Relation type: "organization member of"
Complete triplets: ["cathay pacific", "a substantial shareholder", "cnac"]

Now it's your turn! Please extract the relation trigger from the following text:
Text: "{text}"
Subject, Object (not ordered): "{head}", "{tail}"
Relation type: "{relation}"
Complete triplets:
""".strip()

relation_definition_prompt = """
Define the relationship in a relational triplet extracted from a given text and provide 3 sentence examples of the relationship.
You must generate {k} diverse samples of (relation definition, example) pairs for the relationship.

Example 1:
Text: "he passed away on saturday ."
Triplet: ["he", "passed away on", "saturday"]
Relation type: "person date of death"
Definitions and examples of "passed away on":
Sample 1:
```json
{{
    "definition": "The relationship between a person and the date of their death.",
    "examples": ["he was taken off life support on feb. 14 .", "carolyn goodman , a woman i was privileged to call a friend , died last month at the age of 91 .", "today the nypd upgraded the charges to include murder , in the case of brooklyn gay-bashing/robbery victim michael sandy , who died on friday after being taken off life-support ."]
}}
```
...

Now it's your turn! Please define the relationship in the following relational triplet:
Text: "{text}"
Triplet: {triplet}
Relation type: "{relation}"
Definitions and examples of "{trigger}":
""".strip()

file = "./data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt"

raw_data = []

with open(file) as f:
    for line in f:
        items = line.strip().split('\t')
        raw_data.append(items)

from tqdm import tqdm
import re
import json

with open("./data/CFRLFewRel/relation_name.txt") as f:
    relation_names = f.readlines()

k = 10

# process and save to txt file line by line
with open(f"./data/CFRLFewRel/CFRLdata_10_100_10_5/train_0_{k}.txt", "w") as f:
    for data in tqdm(raw_data):
        text, head, tail = data[2], data[3], data[5]
        relation_name = relation_names[int(data[0])-1]

        prompt = extract_trigger_prompt.format(text=text, head=head, tail=tail, relation=relation_name)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
            ],
            top_p=0.5,
            max_tokens=256,
            stream=False
        )
        content = response.choices[0].message.content.strip()

        try:
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, content)
            triplet = json.loads("[" + matches[0] + "]")
        except Exception as e:
            print(e)
            print(content)
            print(data)
            triplet = [head, "null", tail]

        if triplet[1] != "null":
            prompt = relation_definition_prompt.format(text=text, triplet=json.dumps(triplet), trigger=triplet[1], relation=relation_name, k=k)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                top_p=0.5,
                max_tokens=2048,
                stream=False
            )
            response = response.choices[0].message.content.strip()

            try:
                samples = re.findall(r"```json\n(.*?)\n```", response, re.DOTALL)
                samples = [json.loads(sample) for sample in samples]
                relation_descriptions = [sample["definition"] + " Examples: " + "; ".join(sample["examples"]) for sample in samples]
            except Exception as e:
                print(e)
                print(f"Text: {text}")
                print(f"Relation description response: {response}")
                samples = [{"definition": "null", "examples": ["null"]*3}]*k
                relation_descriptions = [sample["definition"] + " Examples: " + "; ".join(sample["examples"]) for sample in samples]
        else:
            relation_descriptions = ["null"]*k
        
        data = data + [json.dumps(triplet)] + relation_descriptions

        line_to_write = "\t".join(data) + "\n"

        f.write(line_to_write)
