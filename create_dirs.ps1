$dirs = @(
    'C:\ai_soc\app\_template',
    'C:\ai_soc\app\lanes',
    'C:\ai_soc\app\lanes\netflow',
    'C:\ai_soc\app\lanes\siem_edr',
    'C:\ai_soc\app\lanes\cloud_entity',
    'C:\ai_soc\app\lanes\_template',
    'C:\ai_soc\app\core',
    'C:\ai_soc\app\api',
    'C:\ai_soc\app\db',
    'C:\ai_soc\app\execution',
    'C:\ai_soc\app\orchestration',
    'C:\ai_soc\app\llm',
    'C:\ai_soc\app\reporting',
    'C:\ai_soc\app\utils',
    'C:\ai_soc\data\raw\netflow',
    'C:\ai_soc\data\raw\siem_edr',
    'C:\ai_soc\data\raw\cloud_entity',
    'C:\ai_soc\data\processed',
    'C:\ai_soc\data\uploads',
    'C:\ai_soc\data\samples',
    'C:\ai_soc\models\netflow\1.0.0',
    'C:\ai_soc\models\siem_edr\1.0.0',
    'C:\ai_soc\models\cloud_entity\1.0.0',
    'C:\ai_soc\knowledge',
    'C:\ai_soc\outputs\reports',
    'C:\ai_soc\outputs\charts',
    'C:\ai_soc\scripts',
    'C:\ai_soc\tests\contract',
    'C:\ai_soc\tests\core',
    'C:\ai_soc\tests\api',
    'C:\ai_soc\tests\orchestration',
    'C:\ai_soc\tests\execution',
    'C:\ai_soc\frontend\src\components'
)
foreach ($d in $dirs) {
    mkdir $d
    Write-Host "Created $d"
}