# radmerti.github.io

Use the following command to resize images:

```
magick convert -strip -interlace Plane -gaussian-blur 0.05 -quality "85%" -resize "1410x>" rocket-launch.jpg rocket-launch.jpg
```