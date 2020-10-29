if [ $1 == 'r' ]
  then
    if [ $# -eq 3 ];then
      proxychains4 rsync -ztrlvC --progress web_server@10.62.34.36::files/konglingshu/$2 $3
      echo "proxychains4 rsync -ztrlvC --progress web_server@10.62.34.36::files/konglingshu/$2 $3"
    else
      proxychains4 rsync -ztrlvC --progress web_server@10.62.34.36::files/konglingshu/$2 .
      echo "proxychains4 rsync -ztrlvC --progress web_server@10.62.34.36::files/konglingshu/$2 ."
    fi
elif [ $1 == 's' ]; then
  if [ $# -eq 3 ];then
      proxychains4 rsync -ztrlvC --progress $2 web_server@10.62.34.36::files/konglingshu/$3
      echo "proxychains4 rsync -ztrlvC --progress $2 web_server@10.62.34.36::files/konglingshu/$3"
      echo "rsync -avC --progress web_server@10.62.34.36::files/konglingshu/$3/$2 ."
  else
    proxychains4 rsync -ztrlvC --progress $2 web_server@10.62.34.36::files/konglingshu/
    echo "proxychains4 rsync -ztrlvC --progress $2 web_server@10.62.34.36::files/konglingshu/"
    echo "rsync -avc --progress web_server@10.62.34.36::files/konglingshu/$2 ."
  fi
fi
