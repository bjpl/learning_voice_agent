#!/bin/bash
# Generate PWA icons from a source SVG or PNG
# Usage: ./generate-icons.sh source-icon.svg

# Icon sizes needed for PWA
sizes=(72 96 128 144 152 192 384 512)

# Create icons directory if it doesn't exist
mkdir -p icons

# Generate placeholder SVG icons (blue circle with microphone)
for size in "${sizes[@]}"; do
  cat > "icon-${size}x${size}.svg" << EOF
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}">
  <rect width="${size}" height="${size}" rx="$((size/8))" fill="#3b82f6"/>
  <g transform="translate($((size/4)) $((size/4))) scale($((size/512)))">
    <path fill="white" d="M128 0C92.65 0 64 28.65 64 64v128c0 35.35 28.65 64 64 64s64-28.65 64-64V64c0-35.35-28.65-64-64-64zm96 192c0 53.02-42.98 96-96 96s-96-42.98-96-96H0c0 71.18 49.94 130.77 116.8 145.78V384H80v32h96v-32h-36.8V337.78C206.06 322.77 256 263.18 256 192h-32z"/>
  </g>
</svg>
EOF
done

echo "Generated placeholder SVG icons"
echo "For production, replace with actual PNG icons using:"
echo "  convert icon.svg -resize NxN icon-NxN.png"
