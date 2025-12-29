target_user=${SUDO_USER:-$USER}
pgrep -u "$target_user" -af "python|python3|ipython|jupyter|torchrun"
