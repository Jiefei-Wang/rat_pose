# Video conversion script
$sourceDir = "F:\code\pose_track\projects\rat_pose\videos"
$destDir = "F:\code\pose_track\projects\rat_pose\videos_converted"

# Create destination directory if it doesn't exist
if (!(Test-Path $destDir)) {
    New-Item -ItemType Directory -Path $destDir | Out-Null
}

# Videos that are already 1280x720 MP4 - just copy
$copyFiles = @(
    "Camera4_stitched.mp4",
    "RAT 11 FR1.mp4",
    "ai1.mp4",
    "ai2.mp4",
    "ai3.mp4",
    "ai4.mp4",
    "ai5.mp4",
    "ai6.mp4",
    "ai7.mp4",
    "ai8.mp4"
)

Write-Host "=== Copying MP4 files already at 1280x720 ===" -ForegroundColor Green
foreach ($file in $copyFiles) {
    $sourcePath = Join-Path $sourceDir $file
    $destPath = Join-Path $destDir $file
    
    if (Test-Path $sourcePath) {
        Write-Host "Copying: $file"
        Copy-Item $sourcePath $destPath -Force
    } else {
        Write-Host "Skipping (not found): $file" -ForegroundColor Yellow
    }
}

Write-Host "`n=== Converting 1920x1080 videos to 1280x720 MP4 ===" -ForegroundColor Green

# Videos that need conversion (1920x1080 -> 1280x720)
$convertFiles = @(
    "RAT 11 FR1 10-02-25.mkv",
    "RAT 11 FR1 10-03-25.mkv",
    "RAT 2 FR1 10-02-25.mkv",
    "RAT 2 FR1 10-03-25.mkv",
    "RAT 4 FR1 10-02-25.mkv",
    "RAT 4 FR1 10-03-25.mkv",
    "RAT 6 FR1 10-02-25.mkv",
    "RAT 6 FR1 10-03-25.mkv",
    "RAT 8 FR1 10-02-25.mkv",
    "RAT 8 FR1 10-03-25.mkv",
    "2025-03-18 12-58-10.mkv",
    "2025-03-18 12-58-46.mkv",
    "2025-03-18 14-08-10.mkv"
)

foreach ($file in $convertFiles) {
    $sourcePath = Join-Path $sourceDir $file
    $outputName = [System.IO.Path]::GetFileNameWithoutExtension($file) + ".mp4"
    $destPath = Join-Path $destDir $outputName
    
    if (Test-Path $sourcePath) {
        Write-Host "Converting: $file -> $outputName"
        ffmpeg -i $sourcePath -vf scale=1280:720 -c:v h264_nvenc -crf 23 -preset medium -c:a aac -b:a 128k $destPath -y
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Completed: $outputName" -ForegroundColor Green
        } else {
            Write-Host "  ✗ Failed: $file" -ForegroundColor Red
        }
    } else {
        Write-Host "Skipping (not found): $file" -ForegroundColor Yellow
    }
}

Write-Host "`n=== Conversion Complete ===" -ForegroundColor Cyan
Write-Host "Output directory: $destDir"
