import getpass
import netrc
import os
from pathlib import Path

from ._types import Filename

NASA_HOST = "urs.earthdata.nasa.gov"


def setup_nasa_netrc(netrc_file: Filename = "~/.netrc"):
    """Prompt user for NASA username/password, store as attribute of ~/.netrc."""
    netrc_file = Path(netrc_file).expanduser()
    try:
        n = netrc.netrc(netrc_file)
        has_correct_permission = _file_is_0600(netrc_file)
        if not has_correct_permission:
            print(
                "Your ~/.netrc file does not have the correct permissions.\n"
                "Changing permissions to 0600 (read/write for user only)."
            )
            os.chmod(netrc_file, 0o600)
        # Check account exists, as well is having username and password
        _has_nasa_entry = (
            NASA_HOST in n.hosts
            and n.authenticators(NASA_HOST)[0]  # type: ignore
            and n.authenticators(NASA_HOST)[2]  # type: ignore
        )
        if _has_nasa_entry:
            return
    except FileNotFoundError:
        # User doesn't have a netrc file, make one
        print("No ~/.netrc file found, creating one.")
        Path(netrc_file).write_text("")
        n = netrc.netrc(netrc_file)

    username, password = _get_username_pass()
    # Add NASA account to netrc file
    n.hosts[NASA_HOST] = (username, "", password)
    print(f"Saving credentials to {netrc_file} (machine={NASA_HOST}).")
    with open(netrc_file, "w") as f:
        f.write(str(n))
    # Set permissions to 0600 (read/write for user only)
    # https://www.ibm.com/docs/en/aix/7.1?topic=formats-netrc-file-format-tcpip
    os.chmod(netrc_file, 0o600)


def _file_is_0600(filename: Filename):
    """Check that a file has 0600 permissions (read/write for user only)."""
    return oct(Path(filename).stat().st_mode)[-4:] == "0600"


def _get_username_pass():
    """If netrc is not set up, get username/password via command line input."""
    print("Please enter NASA Earthdata credentials to download ASF-hosted data.")
    print("See https://urs.earthdata.nasa.gov/users/new for signup info")
    username = input("Username: ")
    password = getpass.getpass("Password (will not be displayed): ")
    return username, password
