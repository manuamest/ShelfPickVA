# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup.
- Hand dwell time logic to influence bottle probabilities.
- Label-based ROI image naming and implicit loading.
- Redesigned probabilities panel (full-width, square cards).

### Changed
- Updated `app/main.py` to remove dependency on `image_path` in config.
- Updated `app/config_server.py` to generate label-based filenames.
