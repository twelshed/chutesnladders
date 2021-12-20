
for ($td = 100 ; $td -le 400 ; $td+=50)
{
	for ($sig = 1 ; $sig -le 20 ; $sig++)
	{
		"I count $sig $td"
		Start-Process -FilePath "python" -ArgumentList ".\run.py $sig $td" -Wait
	}
}