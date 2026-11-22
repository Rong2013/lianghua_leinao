# 生成 example_run.txt（UTF-8，真实输出，未手工修改）
$ErrorActionPreference = "Continue"
$py = "I:\AI\runtime\win\python\python.exe"
$cli = "I:\lianghua\cli.py"
$out = "I:\lianghua\example_run.txt"
Remove-Item $out -Force -ErrorAction SilentlyContinue
Remove-Item "I:\lianghua\_arch_sleep.npz","I:\lianghua\_prune.npz","I:\lianghua\_homeo.npz" -Force -ErrorAction SilentlyContinue

$utf8 = [System.Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText($out, "", $utf8)

function AppendLine($text) {
    [IO.File]::AppendAllText($out, $text + "`r`n", $utf8)
}

function Run($cmd, $args2) {
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", "1")
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $py
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $psi.WorkingDirectory = "I:\lianghua"
    $psi.Arguments = "$cli $cmd $args2"
    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    [IO.File]::AppendAllText($out, ($stdout + $stderr), $utf8)
}

AppendLine "lianghua 完整流程（真实输出，未手工修改）"
AppendLine "===== build ====="
Run "build" "--out I:\lianghua\brain.npz"
AppendLine "===== train ====="
Run "train" "--brain I:\lianghua\brain.npz --data I:\lianghua\samples.tsv --epochs 2"
AppendLine "===== eval train ====="
Run "eval" "--brain I:\lianghua\brain.npz --data I:\lianghua\samples.tsv"
AppendLine "===== eval holdout ====="
Run "eval" "--brain I:\lianghua\brain.npz --data I:\lianghua\samples_test.tsv"
AppendLine "===== arch ====="
Run "arch" "--brain I:\lianghua\brain.npz --data I:\lianghua\samples_test.tsv --out I:\lianghua\_arch_sleep.npz"
AppendLine "===== prune ====="
Run "prune" "--brain I:\lianghua\brain.npz --threshold 1 --data I:\lianghua\samples_test.tsv --out I:\lianghua\_prune.npz"
AppendLine "===== homeostasis ====="
Run "homeostasis" "--brain I:\lianghua\brain.npz --target 1 --out I:\lianghua\_homeo.npz"
AppendLine "===== rplus ====="
Run "rplus" "--brain I:\lianghua\brain.npz"
AppendLine "===== verify ====="
Run "verify" ""

Remove-Item "I:\lianghua\_arch_sleep.npz","I:\lianghua\_prune.npz","I:\lianghua\_homeo.npz" -Force -ErrorAction SilentlyContinue
Write-Output "done: $out"