import sys
import os


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

filenum = 0
prefix = "test2_"
use_pixeldraw = True
pix_height = 64
image_height = pix_height * 4
num_cuts = 65

for text, init_image_key in texts:
  iteration = 200
  aspect = "square"
  # scale = 4
  # scale = 2
  # seed = 0
  seed = -1
  if seed == 0: seen = None
  # monochrom = False
  monochrom = False
  init_image = templates[init_image_key]

  # Simple setup
  import clipit
  from pixeldrawer import PixelDrawer



  # these are good settings for pixeldraw
  clipit.reset_settings()

  clipit.add_settings(size=[image_height,image_height], pixel_size=[pix_height,pix_height])  
  clipit.add_settings(prompts=text)
  clipit.add_settings(quality="normal", num_cuts=num_cuts) #"better"
  clipit.add_settings(use_pixeldraw=use_pixeldraw)
  clipit.add_settings(iterations=iteration, display_every=10)
  clipit.add_settings(seed=seed)
  clipit.add_settings(do_mono=monochrom)
  clipit.add_settings(init_image=init_image)
  clipit.add_settings(output=f"outputs/loot/{prefix}{filenum:03}.png")
  #clipit.add_settings(target_images='')
  #clipit.add_settings(animation_dir='')
  #clipit.add_settings(video=True)
  #clipit.add_settings(save_every=50)

  settings = clipit.apply_settings()
  clipit.do_init(settings)
  # clear_output()
  clipit.drawer.num_cols = 64
  clipit.drawer.num_rows = 64
  clipit.drawer.end_num_cols = 64
  clipit.drawer.end_num_rows = 64
  clipit.do_run(settings)