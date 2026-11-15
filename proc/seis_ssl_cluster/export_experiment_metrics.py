"""Export retained F3, Parihaka and Volve metrics as portable CSV and HTML."""

from seis_ssl_cluster.results.experiment_catalog import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
	main()
