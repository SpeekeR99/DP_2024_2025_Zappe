BEGIN { FS=OFS="," }
NR==1 { 
	noid=1
	for(i=1; i<=NF; i++) { if ($i == "SecurityID") { col=i; noid=0; } }
	print 
}

NR>1 && ( $col == id || noid==1 )  { print } 
