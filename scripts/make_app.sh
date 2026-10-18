#!/bin/bash
# 얇은 .app 런처 만들기 — 더블클릭으로 대시보드 창과 메뉴바 🪜 를 띄운다.
#
#   bash scripts/make_app.sh            # ~/Applications 에 만든다
#   bash scripts/make_app.sh --here     # 저장소 안에 만든다 (확인용)
#
# **파이썬·torch 를 싸매지 않는다.** 이 저장소의 .venv 를 그대로 부르는 5KB 짜리
# 껍데기다. 자립형 번들(700MB~1GB)은 남에게 배포할 때나 의미가 있고, 그 경우에도
# 키움 계좌·데이터 수집이라는 진짜 벽은 그대로 남는다 — docs/ 의 배포 계획 참조.

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="그리드매매"
DEST="$HOME/Applications"
[[ "${1:-}" == "--here" ]] && DEST="$PROJECT/outputs"

APP="$DEST/$NAME.app"
PY="$PROJECT/.venv/bin/python"

[[ -x "$PY" ]] || { echo "가상환경이 없다: $PY"; exit 1; }
"$PY" -c "import rumps" 2>/dev/null || { echo "rumps 가 없다:  $PY -m pip install rumps"; exit 1; }

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$NAME</string>
    <key>CFBundleDisplayName</key><string>$NAME</string>
    <key>CFBundleIdentifier</key><string>com.capstone.stock.menubar</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundleExecutable</key><string>launcher</string>
    <key>CFBundleIconFile</key><string>icon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
</dict>
</plist>
PLIST

# 런처: 저장소 경로를 빌드 시점에 박는다. 내 기계용 껍데기라 이게 맞다 —
# 배포용이면 경로를 찾아 헤매는 대신 setup.sh 가 다시 만들면 된다.
cat > "$APP/Contents/MacOS/launcher" <<LAUNCH
#!/bin/bash
cd "$PROJECT" || exit 1
export APP_ICON="$APP/Contents/Resources/icon.icns"   # 창 앱이 Dock 아이콘으로 쓴다
# 상단바 🪜 가 없으면 같이 띄운다 — 창을 닫아도 🪜 는 남는다
pgrep -f "[Pp]ython[0-9.]* .*scripts/menubar\\.py\$" >/dev/null ||
  nohup "$PY" "$PROJECT/scripts/menubar.py" >/dev/null 2>&1 &
pgrep -f "[Pp]ython[0-9.]* .*scripts/desktop\\.py\$" >/dev/null && exit 0   # 창이 이미 열려 있다
exec "$PY" "$PROJECT/scripts/desktop.py"
LAUNCH
chmod +x "$APP/Contents/MacOS/launcher"

# 아이콘 — 이모지를 그려 icns 로. macOS 기본 도구만 쓴다(sips·iconutil).
ICONSET="$(mktemp -d)/icon.iconset"
mkdir -p "$ICONSET"
"$PY" - "$ICONSET/icon_512x512.png" <<'PYICON'
"""이모지를 그대로 512px PNG 로. AppKit 은 rumps 가 이미 끌고 오는 pyobjc 에 들어 있다
(matplotlib 은 Apple Color Emoji 를 못 찾는다)."""
import sys
from AppKit import (
    NSAttributedString, NSBitmapImageRep, NSColor, NSFont, NSFontAttributeName,
    NSForegroundColorAttributeName, NSGraphicsContext, NSImage, NSPNGFileType,
)
from Foundation import NSMakeRect, NSMakeSize, NSPoint

SIZE = 512
img = NSImage.alloc().initWithSize_(NSMakeSize(SIZE, SIZE))
img.lockFocus()
NSColor.clearColor().set()
attrs = {
    NSFontAttributeName: NSFont.fontWithName_size_("Apple Color Emoji", 380)
                         or NSFont.systemFontOfSize_(380),
    NSForegroundColorAttributeName: NSColor.blackColor(),
}
text = NSAttributedString.alloc().initWithString_attributes_("\U0001FA9C", attrs)
sz = text.size()
text.drawAtPoint_(NSPoint((SIZE - sz.width) / 2, (SIZE - sz.height) / 2))
rep = NSBitmapImageRep.alloc().initWithFocusedViewRect_(NSMakeRect(0, 0, SIZE, SIZE))
img.unlockFocus()
rep.representationUsingType_properties_(NSPNGFileType, None).writeToFile_atomically_(
    sys.argv[1], True)
PYICON
for s in 16 32 64 128 256; do
  sips -z $s $s "$ICONSET/icon_512x512.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/icon.icns"

echo "만들었다: $APP"
du -sh "$APP"
echo "실행:  open \"$APP\"     (대시보드 창 + 상단 메뉴바 🪜)"
