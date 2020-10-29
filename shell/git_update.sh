push_or_pull='s'
remote_src='origin'
branch_name="ken_dev_temp"

if [ $# -gt 0 ]
    then
        push_or_pull=$1
fi

if [ $# -gt 1 ]
    then
        remote_src=$2
fi

if [ $# -gt 2 ]
    then
        branch_name=$3
fi


git branch
sleep 1s

git status
read -p "the remote_src is: $remote_src and whether Continue? [Y/N]" verify

if [ $verify == 'Y' -o $verify == 'y' -o $verify == '' ]
    then
        read -p "if execute 'git checkout -b $branch_name' command? [Y/N]" checkout
        if [ $checkout == 'Y' -o $checkout == 'y' -o $checkout == '' ]
            then
                git checkout -b $branch_name
        fi

        if [ $push_or_pull == 's' -o $push_or_pull == 'S' ]
            then
                # push
                git add .
                git commit -m 'commited to sync'
                git push $remote_src $branch_name:$branch_name
        elif [ $push_or_pull == 'r' -o $push_or_pull == 'R' ]
            then
                # pull
                echo 'pull from git'
                git pull $remote_src $branch_name:$branch_name
        else
            echo 'do nothing'
        fi
else
    echo 'do nothing'
fi
