#!/bin/bash
echo "Running zip script for hw3..."
echo
echo
if [ -f assignment.py ] && [ -f README ];
	then
		echo "CS Login:" \(e.x.: kjin2\):
		read cslogin
		echo
		rm -f hw3_${cslogin}.zip
		zip -r hw3_${cslogin}.zip . -x "*.git*" "*data*" "*MNIST_DATA*" "*.ipynb_checkpoints*" "*zip.sh" "*requirements.txt" ".env/*" ".DS_Store"
		echo
		echo
		echo "Zip script finished, hand in with handin script."
	else
		echo "Missing required files!"
		echo
		echo "Files required:"
		echo "assignment.py"
		echo "README"
fi
