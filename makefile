gh-page:
	poetry run ghp-import -n -f book/_build/html
	git push -f https://github.com/fdcir3tt/kefir_ajuste.git gh-pages
jp_book:
	poetry run jupyter-book build book/