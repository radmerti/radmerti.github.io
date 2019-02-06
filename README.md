# radmerti.github.io

Use the following command to resize images:

```bash
magick convert -strip -interlace Plane -gaussian-blur 0.05 -quality "85%" -resize "1410x>" rocket-launch.jpg rocket-launch.jpg
```

If a png has a tranparent background you can use the following
command before the above command to remove the transparency:

```bash
magick convert -flatten img1.png img1-white.png
```

Use the following command to create a favicon ico from a png:

```bash
magick convert robot-icon.png -define icon:auto-resize=64,48,32,16 favicon.ico
```