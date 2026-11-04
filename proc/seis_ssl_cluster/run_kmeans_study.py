"""Thin CLI for the matched unordered k-means study."""

from seis_ssl_cluster.clustering.kmeans_study import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
	main()
