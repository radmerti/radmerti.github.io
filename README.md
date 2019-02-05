# radmerti.github.io

Use the following command to resize images:

```
magick convert -strip -interlace Plane -gaussian-blur 0.05 -quality "85%" -resize "1410x>" rocket-launch.jpg rocket-launch.jpg
```

Use the following command to create a favicon ico from a png:

```
magick convert robot-icon.png -define icon:auto-resize=64,48,32,16 favicon.ico
```