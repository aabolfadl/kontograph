<#
.SYNOPSIS
    KONTOGRAPH task runner for Windows.

.DESCRIPTION
    `make` is not present on a stock Windows install, and telling a contributor to
    install GNU Make before they can run the project defeats the point of
    design principle P10. This script exposes the same target names as the
    Makefile and delegates to the same `kg` CLI, so the two cannot drift in
    behaviour - only in syntax.

.EXAMPLE
    .\make.ps1 up
    .\make.ps1 test
    .\make.ps1              # lists targets
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = 'help',

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$Compose = @('compose', '-f', 'docker/compose.yml')

function Invoke-Step {
    param([string]$Exe, [string[]]$Arguments)
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe exited with code $LASTEXITCODE"
    }
}

function Invoke-Kg { param([string[]]$Arguments) Invoke-Step 'uv' (@('run', 'kg') + $Arguments) }
function Invoke-Uv { param([string[]]$Arguments) Invoke-Step 'uv' (@('run') + $Arguments) }
function Invoke-Compose { param([string[]]$Arguments) Invoke-Step 'docker' ($Compose + $Arguments) }

$targets = [ordered]@{
    'help'      = 'Show this help'
    'setup'     = 'Install the workspace into .venv'
    'doctor'    = 'Check this machine can run the stack'
    'clean'     = 'Remove generated data and artifacts'
    'up'        = 'Start the local stack (Redpanda, object store, Redis, MLflow)'
    'down'      = 'Stop the stack, keeping volumes'
    'nuke'      = 'Stop the stack and delete its volumes'
    'logs'      = 'Tail stack logs'
    'ps'        = 'Show stack status'
    'flink-up'  = 'Start the optional Flink profile'
    'obs-up'    = 'Start the optional observability profile'
    'test'      = 'Run the full test suite'
    'test-fast' = 'Run tests, skipping anything needing the stack'
    'lint'      = 'Lint without modifying files'
    'fmt'       = 'Auto-format and fix'
    'typecheck' = 'Static type check'
    'contracts' = 'Fail if a contract change would break consumers'
    'ci'        = 'Everything CI runs'
    'sim'       = 'Generate the synthetic payments dataset'
    'register'  = 'Register the Avro contracts with the Schema Registry'
    'replay'    = 'Replay the dataset onto the payments topic'
    'ingest'    = 'Load payments into the bronze Iceberg layer'
    'promote'   = 'Promote bronze to silver through the quality gate'
    'marts'     = 'Export to DuckDB and build the gold marts with dbt'
    'features'  = 'Build the training feature table from silver'
    'verify-pit' = 'Prove the backends agree and nothing sees the future'
    'pipeline'  = 'Simulator -> bronze -> silver -> gold -> features'
    'dagster'   = 'Open the Dagster asset graph'
    'eval'      = 'Regenerate every published number from artifacts'
    'serve'     = 'Run the decision gateway locally'
    'loadtest'  = 'k6 load test; fails if p99 exceeds the SLA'
    'govern'    = 'Regenerate model cards, lineage and the decision-log audit'
}

switch ($Target) {
    'help' {
        Write-Host ''
        Write-Host 'KONTOGRAPH' -ForegroundColor Cyan
        Write-Host '  Usage: .\make.ps1 <target>' -ForegroundColor DarkGray
        Write-Host ''
        foreach ($key in $targets.Keys) {
            Write-Host ('  {0,-12} {1}' -f $key, $targets[$key])
        }
        Write-Host ''
    }

    'setup' {
        Invoke-Step 'uv' @('sync', '--all-packages')
        Write-Host "Run '.\make.ps1 doctor' to check the environment." -ForegroundColor Green
    }
    'doctor' { Invoke-Kg @('doctor') }

    'clean' {
        foreach ($path in 'data', 'artifacts', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.hypothesis') {
            if (Test-Path $path) { Remove-Item -Recurse -Force $path }
        }
        Get-ChildItem -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host 'Cleaned.' -ForegroundColor Green
    }

    'up' {
        Invoke-Compose @('up', '-d')
        Write-Host ''
        Write-Host '  Redpanda console  http://localhost:8080'
        Write-Host '  Object store      http://localhost:8888  (S3 API on :9000, kontograph / kontograph)'
        Write-Host '  MLflow            http://localhost:5000'
    }
    'down' { Invoke-Compose @('down') }
    'nuke' { Invoke-Compose @('down', '-v') }
    'logs' { Invoke-Compose @('logs', '-f', '--tail=100') }
    'ps'   { Invoke-Compose @('ps') }

    'flink-up' {
        Invoke-Step 'docker' @('compose', '-f', 'docker/compose.yml', '-f', 'docker/compose.flink.yml', '--profile', 'flink', 'up', '-d')
    }
    'obs-up' {
        Invoke-Step 'docker' @('compose', '-f', 'docker/compose.yml', '-f', 'docker/compose.obs.yml', '--profile', 'obs', 'up', '-d')
    }

    'test'      { Invoke-Uv (@('pytest') + $Rest) }
    'test-fast' { Invoke-Uv @('pytest', '-m', 'not integration and not slow') }
    'lint' {
        Invoke-Uv @('ruff', 'check', '.')
        Invoke-Uv @('ruff', 'format', '--check', '.')
    }
    'fmt' {
        Invoke-Uv @('ruff', 'format', '.')
        Invoke-Uv @('ruff', 'check', '--fix', '.')
    }
    'typecheck' { Invoke-Uv @('mypy', 'packages') }
    'contracts' { Invoke-Kg @('contracts', 'check') }
    'ci' {
        foreach ($step in 'lint', 'typecheck', 'contracts', 'test') {
            & $PSCommandPath $step
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }

    'sim'      { Invoke-Kg (@('sim', 'generate') + $Rest) }
    'register' { Invoke-Kg @('stream', 'register') }
    'replay'   { Invoke-Kg (@('stream', 'replay') + $Rest) }
    'ingest'   { Invoke-Kg (@('lake', 'ingest') + $Rest) }
    'promote'  { Invoke-Kg @('lake', 'promote') }
    'marts' {
        Invoke-Kg @('lake', 'export')
        Invoke-Kg @('lake', 'dbt', 'build')
    }
    'features'   { Invoke-Kg (@('features', 'build') + $Rest) }
    'verify-pit' { Invoke-Kg (@('features', 'verify') + $Rest) }
    'pipeline' {
        foreach ($step in 'sim', 'ingest', 'promote', 'marts', 'features') {
            & $PSCommandPath $step
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
    'dagster'  { Invoke-Uv @('dagster', 'dev', '-f', 'orchestration/dagster/definitions.py') }
    'eval'     { Invoke-Kg (@('eval', 'report') + $Rest) }
    'serve'    { Invoke-Kg (@('serve', 'run') + $Rest) }
    'loadtest' { Invoke-Kg (@('serve', 'loadtest') + $Rest) }
    'govern'   { Invoke-Kg (@('gov', 'all') + $Rest) }

    default {
        Write-Host "Unknown target '$Target'." -ForegroundColor Red
        Write-Host "Run '.\make.ps1' with no arguments to list targets." -ForegroundColor DarkGray
        exit 1
    }
}
