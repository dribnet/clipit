import sys
import os
import clipit

templates = {
    'amulet': 'templates/amulet.png',
    'armor': 'templates/armor.png',
    'belt': 'templates/belt.png',
    'boots': 'templates/boots.png',
    'gloves': 'templates/gloves.png',
    'greaves': 'templates/greaves.png',
    'hood': 'templates/hood.png',
    'katana': 'templates/katana.png',
    'necklace': 'templates/necklace.png',
    'quarterstaff': 'templates/quarterstaff.png',
    'ring': 'templates/ring.png',
    'robe': 'templates/robe.png',
    'wand': 'templates/wand.png',
}

# this is an example of how different items could have different settings
smoothness_settings = {
  'robe'  : 100,
  'gloves': 100,
  'default': 500
}

texts = [
  ('hard leather armor #pixelart', 'armor'),
  ('"death root" ornate greaves of skill #pixelart', 'greaves'),
  ('studded leather gloves #pixelart', 'gloves'),
  ('divine hood #pixelart', 'hood'),
  ('necklace of enlightenment #pixelart', 'necklace'),
  ('gold ring #pixelart', 'ring'),
  ('hard leather belt #pixelart', 'belt'),
  ('"grim shout" grave wand of skill +1 #pixelart', 'wand'),
  ('"ghoul sun" silk robe of fury #pixelart', 'robe'),
  ('"pandemonium shout" demonhide boots of brilliance +1 #pixelart', 'boots'),
  ('gloves #pixelart', 'gloves'),
  ('amulet #pixelart', 'amulet'),
  ('titanium ring #pixelart', 'ring'),
  ('plated belt #pixelart', 'belt'),
  ('bone wand #pixelart', 'wand'),
  ('ornate greaves of anger #pixelart', 'greaves'),
  ('dragonskin gloves #pixelart', 'gloves'),
  ('necklace #pixelart', 'necklace'),
  ('gold ring of titans #pixelart', 'ring'),
  ('ornate belt of detection #pixelart', 'belt'),
  ('katana #pixelart', 'katana'),
  ('ornate greaves #pixelart', 'greaves'),
  ('leather gloves of perfection #pixelart', 'gloves'),
  ('silk hood #pixelart', 'hood'),
  ('heavy belt #pixelart', 'belt'),
  ('holy greaves #pixelart', 'greaves'),
  ('hard leather gloves #pixelart', 'gloves'),
  ("dragon's crown of perfection #pixelart", 'crown'),
  ('dragonskin armor #pixelart', 'armor'),
  ('dragonskin boots #pixelart', 'boots'),
  ('heavy gloves of titans #pixelart', 'gloves'),
  ('hood #pixelart', 'hood'),
  ('necklace of detection #pixelart', 'necklace'),
  ('silver ring #pixelart', 'ring'),
  ('leather belt #pixelart', 'belt'),
  ('"tempest peak" greaves of enlightenment +1 #pixelart', 'greaves'),
  ('gauntlets #pixelart', 'gauntlet'),
  ('platinum ring #pixelart', 'ring'),
  ('"skull bite" hard leather boots of reflection +1 #pixelart', 'boots'),
  ('wool gloves of skill #pixelart', 'gloves'),
  ('war belt of perfection #pixelart', 'belt'),
  ('ghost wand #pixelart', 'wand'),
  ('"soul glow" studded leather armor of rage #pixelart', 'armor'),
  ('silk gloves #pixelart', 'gloves'),
  ('linen hood of fury #pixelart', 'hood'),
  ('gold ring of anger #pixelart', 'ring'),
  ('mesh belt #pixelart', 'belt'),
  ('robe #pixelart', 'robe'),
  ('wool gloves #pixelart', 'gloves'),
  ('"havoc sun" amulet of reflection #pixelart', 'amulet'),
  ('studded leather belt #pixelart', 'belt'),
  ('studded leather boots #pixelart', 'boots'),
  ('ornate gauntlets of vitriol #pixelart', 'gauntlet'),
  ('demon crown #pixelart', 'crown'),
  ('bronze ring #pixelart', 'ring'),
  ('hard leather boots #pixelart', 'boots'),
  ('gold ring of rage #pixelart', 'ring'),
  ('ornate belt #pixelart', 'belt'),
  ('"kraken moon" hard leather armor of skill +1 #pixelart', 'armor'),
  ('heavy boots of protection #pixelart', 'boots'),
  ('linen robe of rage #pixelart', 'robe'),
  ('ring mail #pixelart', 'ring'),
  ('chain boots of giants #pixelart', 'boots'),
  ('studded leather armor #pixelart', 'armor'),
  ('hard leather boots of vitriol #pixelart', 'boots'),
  ("dragon's crown of titans #pixelart", 'crown'),
  ('leather armor #pixelart', 'armor'),
  ('ornate gauntlets #pixelart', 'gauntlet'),
  ('demonhide boots #pixelart', 'boots'),
  ('divine gloves #pixelart', 'gloves'),
  ('"mind shout" linen robe of protection +1 #pixelart', 'robe'),
  ('"brimstone grasp" hard leather boots of rage +1 #pixelart', 'boots'),
  ('linen robe #pixelart', 'robe'),
  ('linen gloves #pixelart', 'gloves'),
  ('silk robe #pixelart', 'robe'),
  ('dragonskin gloves of power #pixelart', 'gloves'),
  ('gold ring of perfection #pixelart', 'ring'),
  ('leather armor of perfection #pixelart', 'armor'),
  ('"rage moon" silk gloves of detection +1 #pixelart', 'gloves'),
  ('quarterstaff #pixelart', 'quarterstaff'),
  ('divine robe of rage #pixelart', 'robe'),
  ('leather boots #pixelart', 'boots'),
  ('leather belt of the fox #pixelart', 'belt'),
  ('dragonskin armor of giants #pixelart', 'armor'),
  ('heavy boots of detection #pixelart', 'boots'),
  ('"pain roar" war belt of power #pixelart', 'belt'),
  ('"pandemonium grasp" hard leather armor of giants +1 #pixelart', 'armor'),
  ('heavy boots #pixelart', 'boots'),
  ('gauntlets of the fox #pixelart', 'gauntlet'),
]

def main():
  num_cuts = 96
  pix_height = 64
  image_height = pix_height * 4
  iterations = 100
  num_cuts = 96
  prefix = "settings1"

  which_item = int(sys.argv.pop())
  text, init_image_key = texts[which_item]
  # setup defaults for loot (ARGV will still override)
  clipit.reset_settings()
  clipit.add_settings(size=[image_height,image_height], pixel_size=[pix_height,pix_height])
  clipit.add_settings(quality="better", num_cuts=num_cuts) #"better"
  clipit.add_settings(use_pixeldraw=True)
  clipit.add_settings(iterations=iterations, save_every=10)
  clipit.add_settings(prompts=text)
  clipit.add_settings(init_image=templates[init_image_key])
  clipit.add_settings(output=f"outputs/loot/{prefix}_{(which_item+1):03}.png")
  if init_image_key in smoothness_settings:
    clipit.add_settings(enforce_smoothness=smoothness_settings[init_image_key])
  elif 'default' in smoothness_settings:
    clipit.add_settings(enforce_smoothness=smoothness_settings['default'])
  settings = clipit.apply_settings()
  clipit.do_init(settings)
  clipit.do_run(settings)

if __name__ == '__main__':
    main()
