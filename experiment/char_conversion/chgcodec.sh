#!/bin/bash
while getopts "edf:" option; do
case ${option} in
e ) #For option e -- encode from utf-8 to windows-1252
    for filename in $(ls ../*.m); do
	echo "${filename}"
	VAR1=$(file -b $filename | gawk '{print $1;}')
	echo $VAR1
	VAR2='CP1252'
	if [ "$VAR1" = "UTF-8" ]; then
		echo "Encoding file $filename to Win-1252"
		python SAT_utf8_to_win1252.py $filename
		rm $filename
		mv ${filename}_win $filename
	else
	    echo "Not encoding file $filename to WIN-1252"
	fi
done
;;
d ) #For option d -- decode to utf-8
for filename in $(ls ../*.m); do
        echo $filename
	VAR1=$(file -b $filename | gawk '{print $1;}')
	echo $VAR1
	if [ "$VAR1" = "UTF-8" ]; then
		echo 'skipping'
	elif [ "$VAR1" = "ASCII" ]; then
		echo 'skipping'
	else
		echo "Decoding file $filename to UTF-8"
		python SAT_win1252_to_utf8.py  $filename
		rm $filename
		mv ${filename}_utf8 $filename		
	fi
done
;;
\? ) #For invalid option
echo "You have to use: [-d], [-e] or [-f filename] "
;;
f )
    echo $OPTARG
    VAR1=$(file -b $OPTARG | gawk '{print $1;}')
    if [ "$VAR1" = "UTF-8" ]; then
	  echo "Encoding file $OPTARG to WIN1252"
	  python SAT_utf8_to_win1252.py $OPTARG
    else
	  echo "Decoding file $OPTARG to UTF-8"
	  python SAT_win1252_to_utf8.py  $OPTARG	
    fi
    
;;
esac
done
