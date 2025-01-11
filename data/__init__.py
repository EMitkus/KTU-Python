# Importuojamos funkcijos iš mysql_handler modulio
from .mysql_handler import save_to_mysql, load_from_mysql

# Nurodomi moduliai ir funkcijos, kurie bus pasiekiami naudojant `from data import *`
__all__ = ['save_to_mysql', 'load_from_mysql']