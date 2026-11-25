/*
 * CrystalX Demo -- the application's main executable.
 *
 * A deliberately tiny native program: it finds its own bundle and hands over to
 * Contents/Resources/launch.sh, where the start logic lives. Being a native arm64 executable
 * (not a script) means macOS never treats the application as Intel code, so no Rosetta 2
 * prompt can appear for it. It starts no network activity and changes no system setting.
 *
 * Build (no Apple SDK needed):  python -m ziglang cc -target aarch64-macos.13.0 -O2 \
 *                                  -o "CrystalX Demo" launcher.c
 * or on a Mac with the Command Line Tools:  clang -arch arm64 -mmacosx-version-min=13.0 -O2 ...
 * The same file builds on Linux for the stand-in acceptance test of the bundle layout.
 */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

static void tell(const char *text) {
#ifdef __APPLE__
    char script[1024];
    pid_t pid;
    snprintf(script, sizeof script, "display alert \"CrystalX Demo\" message \"%s\" as critical", text);
    pid = fork();
    if (pid == 0) {
        execl("/usr/bin/osascript", "osascript", "-e", script, (char *)NULL);
        _exit(127);
    }
    if (pid > 0) {
        int status;
        waitpid(pid, &status, 0);
    }
#endif
    fprintf(stderr, "CrystalX Demo: %s\n", text);
}

static int executable_path(char *out, size_t size) {
    char raw[PATH_MAX];
#ifdef __APPLE__
    uint32_t length = sizeof raw;
    if (_NSGetExecutablePath(raw, &length) != 0) return -1;
#else
    ssize_t length = readlink("/proc/self/exe", raw, sizeof raw - 1);
    if (length <= 0) return -1;
    raw[length] = '\0';
#endif
    if (realpath(raw, out) == NULL) return -1;
    (void)size;
    return 0;
}

int main(void) {
    char exe[PATH_MAX], script[PATH_MAX];
    char *slash;
    if (executable_path(exe, sizeof exe) != 0) {
        tell("Cannot locate the application bundle.");
        return 1;
    }
    /* exe is .../CrystalX Demo.app/Contents/MacOS/CrystalX Demo: strip two components. */
    for (int i = 0; i < 2; i++) {
        slash = strrchr(exe, '/');
        if (slash == NULL || slash == exe) {
            tell("The application bundle has an unexpected layout.");
            return 1;
        }
        *slash = '\0';
    }
    if (snprintf(script, sizeof script, "%s/Resources/launch.sh", exe) >= (int)sizeof script) {
        tell("The application's folder path is too long.");
        return 1;
    }
    char *const args[] = {"/bin/sh", script, NULL};
    execv("/bin/sh", args);
    tell("Cannot run /bin/sh to start CrystalX Demo.");
    return 1;
}
