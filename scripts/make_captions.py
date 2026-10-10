"""Write the 30 caption files (name.txt) for the LoRA training set.

Each caption starts with the trigger word `stardew_pixel_art`, followed by
Danbooru-style tags describing the NPC portrait. Pair each .txt with the
matching name.png portrait, zip the folder, and upload it in
notebooks/01_train_lora.ipynb.

Usage:
    python scripts/make_captions.py --out data/train
"""
import argparse
import os

tags_data = {
    "Clint_1": "stardew_pixel_art, Clint, 1boy, solo, portrait, brown hair, short hair, messy hair, facial hair, beard, mustache, grey sweater, brown apron, neutral expression, white background",
    "Clint_2": "stardew_pixel_art, Clint, 1boy, solo, portrait, brown hair, short hair, messy hair, facial hair, beard, mustache, grey sweater, brown apron, smiling, closed eyes, white background",
    "Clint_3": "stardew_pixel_art, Clint, 1boy, solo, portrait, brown hair, short hair, messy hair, facial hair, beard, mustache, grey sweater, brown apron, sad, looking away, white background",
    "Dobson_1": "stardew_pixel_art, Dobson, 1boy, solo, portrait, brown hair, parted hair, glasses, blue jacket, dark vest, holding phone, neutral expression, white background",
    "Dobson_2": "stardew_pixel_art, Dobson, 1boy, solo, portrait, brown hair, parted hair, glasses, blue jacket, dark vest, holding phone, smiling, white background",
    "Dobson_3": "stardew_pixel_art, Dobson, 1boy, solo, portrait, brown hair, parted hair, glasses, blue jacket, dark vest, holding phone, sad, frowning, white background",
    "Jas_1": "stardew_pixel_art, Jas, 1girl, solo, portrait, purple hair, twin buns, large green bow, hair ornament, purple dress, smiling, blue eyes, white background",
    "Jas_2": "stardew_pixel_art, Jas, 1girl, solo, portrait, purple hair, twin buns, large green bow, hair ornament, purple dress, smiling, closed eyes, white background",
    "Jas_3": "stardew_pixel_art, Jas, 1girl, solo, portrait, purple hair, twin buns, large green bow, hair ornament, purple dress, sad, frowning, blue eyes, white background",
    "Jodi_1": "stardew_pixel_art, Jodi, 1girl, solo, portrait, reddish-brown hair, side braid, braided hair, blue eyes, purple shirt, smiling, white background",
    "Jodi_2": "stardew_pixel_art, Jodi, 1girl, solo, portrait, reddish-brown hair, side braid, braided hair, purple shirt, hand covering mouth, smiling, closed eyes, laughing, white background",
    "Jodi_3": "stardew_pixel_art, Jodi, 1girl, solo, portrait, reddish-brown hair, side braid, braided hair, blue eyes, purple shirt, sad, worried, frowning, white background",
    "Lewis_1": "stardew_pixel_art, Lewis, 1boy, solo, portrait, grey hair, grey mustache, facial hair, brown flat cap, hat, green shirt, suspenders, yellow tie, neutral expression, white background",
    "Lewis_2": "stardew_pixel_art, Lewis, 1boy, solo, portrait, grey hair, grey mustache, facial hair, brown flat cap, hat, green shirt, suspenders, yellow tie, laughing, smiling, open mouth, closed eyes, white background",
    "Lewis_3": "stardew_pixel_art, Lewis, 1boy, solo, portrait, grey hair, grey mustache, facial hair, brown flat cap, hat, green shirt, suspenders, yellow tie, sad, disappointed, looking away, white background",
    "Marnie_1": "stardew_pixel_art, Marnie, 1girl, solo, portrait, brown hair, long hair, wavy hair, brown eyes, green dress, orange scarf, neutral expression, slight smile, white background",
    "Marnie_2": "stardew_pixel_art, Marnie, 1girl, solo, portrait, brown hair, long hair, wavy hair, brown eyes, green dress, orange scarf, smiling, happy, white background",
    "Marnie_3": "stardew_pixel_art, Marnie, 1girl, solo, portrait, brown hair, long hair, wavy hair, brown eyes, green dress, orange scarf, sad, worried, looking away, white background",
    "Maru_Hospital_1": "stardew_pixel_art, Maru, 1girl, solo, portrait, dark skin, dark red hair, short hair, glasses, white nurse cap, nurse uniform, cross symbol, neutral expression, slight smile, white background",
    "Maru_Hospital_2": "stardew_pixel_art, Maru, 1girl, solo, portrait, dark skin, dark red hair, short hair, glasses, white nurse cap, nurse uniform, cross symbol, laughing, big smile, open mouth, closed eyes, white background",
    "Maru_Hospital_3": "stardew_pixel_art, Maru, 1girl, solo, portrait, dark skin, dark red hair, short hair, glasses, white nurse cap, nurse uniform, cross symbol, sad, worried, looking down, white background",
    "Pam_1": "stardew_pixel_art, Pam, 1girl, solo, portrait, older woman, blonde hair, curly hair, voluminous hair, purple eyeshadow, makeup, pink jacket, neutral expression, white background",
    "Pam_2": "stardew_pixel_art, Pam, 1girl, solo, portrait, older woman, blonde hair, curly hair, voluminous hair, purple eyeshadow, makeup, pink jacket, smiling, closed eyes, laughing, white background",
    "Pam_3": "stardew_pixel_art, Pam, 1girl, solo, portrait, older woman, blonde hair, curly hair, voluminous hair, purple eyeshadow, makeup, pink jacket, sad, annoyed, looking away, white background",
    "Pierre_1": "stardew_pixel_art, Pierre, 1boy, solo, portrait, brown hair, parted hair, glasses, blue tinted glasses, brown jacket, blue shirt, neutral expression, slight smile, white background",
    "Pierre_2": "stardew_pixel_art, Pierre, 1boy, solo, portrait, brown hair, parted hair, glasses, blue tinted glasses, brown jacket, blue shirt, smiling, open mouth, happy, white background",
    "Pierre_3": "stardew_pixel_art, Pierre, 1boy, solo, portrait, brown hair, parted hair, glasses, blue tinted glasses, brown jacket, blue shirt, sad, frowning, looking down, white background",
    "Robin_1": "stardew_pixel_art, Robin, 1girl, solo, portrait, orange hair, ponytail, tied hair, green eyes, yellow jacket, hoodie, neutral expression, slight smile, white background",
    "Robin_2": "stardew_pixel_art, Robin, 1girl, solo, portrait, orange hair, ponytail, tied hair, green eyes, yellow jacket, hoodie, laughing, smiling, open mouth, closed eyes, white background",
    "Robin_3": "stardew_pixel_art, Robin, 1girl, solo, portrait, orange hair, ponytail, tied hair, green eyes, yellow jacket, hoodie, sad, looking away, white background"
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/train", help="folder that holds the portrait images")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for filename, tags in tags_data.items():
        with open(os.path.join(args.out, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.write(tags)
    print(f"Wrote {len(tags_data)} caption files to {args.out}")


if __name__ == "__main__":
    main()
