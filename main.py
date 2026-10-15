import sys

DEPRECATION = (
    "The ALchemist desktop app is deprecated and will be removed in a future "
    "release. Use the web app instead: run alchemist-web and open http://localhost:8000."
)


def main():
    print(DEPRECATION, file=sys.stderr)
    try:
        from ui.ui import ALchemistApp
    except ImportError as exc:
        print(
            f"The desktop app needs optional packages that are not installed ({exc}).\n"
            'Install them with: pip install "alchemist-nrel[desktop]"',
            file=sys.stderr,
        )
        sys.exit(1)
    app = ALchemistApp()
    app.mainloop()


if __name__ == '__main__':
    main()
