gh-page:
	poetry run ghp-import -n -p -f book/_build/html
	
jp_book:
	poetry run jupyter-book build book/